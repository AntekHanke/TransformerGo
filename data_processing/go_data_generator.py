import os
from typing import Dict, Any, Optional, List, Type, Union, Tuple

import random
import numpy as np

import pandas as pd
import sente
import torch
from collections import namedtuple

#from matplotlib import pyplot as plt

#from configures.global_config import MAX_GAME_LENGTH
from data_processing.go_tokenizer import GoTokenizer
from data_structures.go_data_structures import GoImmutableBoard, GoMetadata, GoTransition, GoOneGameData
from utils.data_utils import get_split#, immutable_boards_to_img, RESULT_TO_WINNER
from metric_logging import log_value, log_object

#from data_processing.probability_subgoal_selector_tools import prob_table_for_diff_n, prob_select_function
#from policy.chess_policy import LCZeroPolicy

# TODO: fill in fields
GoGameMetadata = namedtuple("GameMetadata", "game_id, winner, result")


class GoFilter:
    """Filters games and transitions to use in training."""

    def use_game(self, game_metadata: GoMetadata) -> bool:
        """Decides whether a game is useful."""
        raise NotImplementedError

    def use_transition(self, transition: GoTransition, one_game_data: GoOneGameData) -> bool:
        """Decides whether a transition is useful."""
        raise NotImplementedError


class NoFilter(GoFilter):
    """Accepts every game and transition"""

    def use_game(self, game_metadata: GoMetadata) -> bool:
        return True

    def use_transition(self, transition: GoTransition, one_game_data: GoOneGameData) -> bool:
        return True


class ResultFilter(GoFilter):
    """Filters games that have a winner or looser. Filters transitions that were played by the winner or looser."""

    def __init__(
        self,
        winner_or_looser: str = "winner",
    ) -> None:
        assert winner_or_looser in ["winner", "loser"], "winner_or_looser must be 'winner' or 'loser'"
        self.winner_or_looser = winner_or_looser

    def use_game(self, game_metadata: GoMetadata) -> bool:
        take_game: bool = False
        try:
            x = game_metadata.RE #If result exists, it must be a win or loose in Go
            take_game: bool = True
        except Exception as e:
            print(f"Error: {e}")
            print("\n")
            print("Can't find inforations about game result. Return False.")
        return take_game

    def use_transition(self, transition: GoTransition, one_game_data: GoOneGameData) -> bool:

        who_won = one_game_data.metadata.RE[0]
        if(who_won == 'B'):
            player_won = sente.BLACK
        else:
            player_won = sente.WHITE

        if self.winner_or_looser == "winner":
            return transition.immutable_board.active_player == player_won
        elif self.winner_or_looser == "loser":
            return transition.immutable_board.active_player != player_won


class RankFilter(GoFilter):
    def __init__(self, rank_threshold: int = 1): #1d = 1, 9d = 9, 1p = 11, 5k = -5
        self.rank_threshold = rank_threshold

    def use_game(self, game_metadata: GoMetadata) -> bool: #Rank is in the form of 3d, 5k, 2p etc.
        take_game: bool = False
        try:
            w_first_digit = game_metadata.WR[0]
            w_rank_type = game_metadata.WR[-1]
            if(w_rank_type == 'd'):
                w_rank = w_first_digit*1
            elif(w_rank_type == 'p'):
                w_rank = w_first_digit+10
            elif(w_rank_type == 'k'):
                w_rank = -w_first_digit

            b_first_digit = game_metadata.BR[0]
            b_rank_type = game_metadata.BR[-1]
            if(b_rank_type == 'd'):
                b_rank = b_first_digit
            elif(b_rank_type == 'p'):
                b_rank = b_first_digit+10
            elif(w_rank_type == 'k'):
                b_rank = -b_first_digit


            take_game = min(b_rank, w_rank) >= self.rank_threshold
        except Exception as e:
            print(f"Error: {e}")
            print("\n")
            print("Can't find inforations about ranks. Return False.")
        return take_game

    def use_transition(self, transition: GoTransition, one_game_data: GoOneGameData) -> bool:
        """Decides whether a transition is useful."""
        raise NotImplementedError


class GoDataset(torch.utils.data.Dataset):
    """Used by Pytorch DataLoader to get batches of data."""

    def __init__(self, data: Dict):
        self.data = data

    def __getitem__(self, idx: int):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


class GoDataProvider:
    """General class for providing data for training and evaluation."""

    def get_train_set_generator(self) -> GoDataset:
        raise NotImplementedError

    def get_eval_set_generator(self) -> GoDataset:
        raise NotImplementedError


class GoGamesDataGenerator(GoDataProvider):
    """Reads PGN file and creates data."""

    def __init__(
        self,
        sgf_file: Optional[str] = None,
        sgf_files: Optional[str] = None,
        go_filter: Optional[Type[GoFilter]] = None,
        p_sample: Optional[float] = None,
        max_games: Optional[int] = None,
        train_eval_split: float = 0.95,
        do_sample_finish: bool = True,
        log_samples_limit: Optional[int] = None,
        log_stats_after_n_games: int = 1000,
        p_log_sample: float = 0.01,
        only_eval: bool = False,
        save_path_to_train_set: Optional[str] = None,
        save_path_to_eval_set: Optional[str] = None,
        save_filtered_data: Optional[str] = None,
        save_data_every_n_games: int = 1000,
    ):
        self.size_of_computed_dataset: int = 0
        self.anchor = os.path.dirname(sgf_files)
        print("Anchor is: ", self.anchor)

        if(sgf_files[-3:]=="aio"):
            self.aio = True
            print("AIO mode enabled")
            self.allgames = []
            self.name_of_sgf_file: str = os.path.basename(sgf_files)[:-4]
            with open(sgf_files, encoding="utf8") as f:
                self.sgf_lines = f.readlines()
            for line in self.sgf_lines:
                if(line[0] == ';'):
                    self.allgames.append('(;BR[]WR[]KM[]RE[]'+line)
            #print("Allgames: ", self.allgames)
        else:
            self.aio = False
            with open(sgf_files, encoding="utf8") as f:
                self.sgf_files_directories = f.readlines()
            self.sgf_file = self.sgf_files_directories[0]
            if self.sgf_files_directories is not None:
                self.path_to_sgf_file = sgf_file
                self.name_of_sgf_file: str = os.path.basename(sgf_files)[:-4]
                #self.sgf_database = open(self.path_to_sgf_file, errors="ignore")


        if go_filter is None:
            go_filter = NoFilter()
        self.go_filter = go_filter
        assert go_filter is not None, "Go filter must be specified"

        print(100 * "=")
        print(
            f"Name of used filter: {type(self.go_filter)}." f" Atributs of used filter: {self.go_filter.__dict__}"
        )
        print(100 * "=")

        self.p_sample = p_sample
        self.max_games = max_games
        self.train_eval_split = train_eval_split
        self.do_sample_finish = do_sample_finish
        self.log_samples_limit = log_samples_limit
        self.log_stats_after_n_games = log_stats_after_n_games
        self.p_log_sample = p_log_sample
        self.only_eval = only_eval

        self.train_data_queue: Dict = dict()
        self.eval_data_queue: Dict = dict()
        self.logged_samples: int = 0
        self.n_games: int = 0
        self.data_constructed: bool = False

        self.save_path_to_train_set = save_path_to_train_set
        self.save_path_to_eval_set = save_path_to_eval_set
        self.save_filtered_data = save_filtered_data
        self.save_data_every_n_games = save_data_every_n_games

        #self.lc_zero_policy = LCZeroPolicy()

    def next_game_to_raw_data(self) -> Optional[GoOneGameData]:
        """
        Function takes the next game from the set of chess games, checks if it passes through the filter used and
        return information about game chass.
        :return: OneGameData contains inforamtion about chess game.
        """

        if (self.aio):
            #print("Loading from sgf: ", self.allgames[self.n_games])
            current_game = sente.sgf.loads(self.allgames[self.n_games])
        else:
            sgf_dir = os.path.normpath(os.path.join(self.anchor, self.path_to_sgf_file))
            if(sgf_dir[-3:]!="sgf"):
                sgf_dir += ".sgf"
            current_game = sente.sgf.load(sgf_dir, disable_warnings=True)

            ### The following few lines are required to fix the handicap errors of Sente SGF loader
            sgf = sente.sgf.dumps(current_game)
            sgf = re.sub("([a-z])\];AB\[", '\\1];W[];AB[', sgf)
            sgf = re.sub("AB\[", 'B[', sgf)
            #print(f"in dir {sgf_dir}, sgf: {sgf}")
            current_game = sente.sgf.loads(sgf)
            #print("properties: ", current_game.get_properties())

        if current_game is None:  # Condition is met if there are no more games in the dataset.
            return None

        go_metadata: GoMetadata = GoMetadata(**current_game.get_properties())
        #print("go_metadata.RE: ", go_metadata.RE)
        transitions: List[GoTransition] = []

        if self.go_filter.use_game(go_metadata):
            transitions, final_pgn_board, is_game_over = self.moves_to_transitions(
                initial_immutable_board=None, moves=current_game.get_default_sequence()
            )

            # if not is_game_over:
            #     lc_zero_transitions, result = self.finish_game(
            #         pgn_final_board=final_pgn_board, move_num=len(transitions)
            #     )
            #
            #     transitions.extend(lc_zero_transitions)
            #     chess_metadata.Result = result

        return GoOneGameData(go_metadata, transitions)

    def moves_to_transitions(self, moves, initial_immutable_board = None) -> Tuple[List[GoTransition], GoImmutableBoard, bool]:
        if initial_immutable_board is None:
            game = sente.Game()
        else:
            #board = initial_immutable_board.to_game()
            pass

        transitions: List[GoTransition] = []
        for move in enumerate(moves):
            move_num, go_move = move
            try:
                if not(go_move == sente.Move(19,19,sente.WHITE) and move_num < 20):
                    transitions.append(GoTransition(GoImmutableBoard.from_game(game), (go_move.get_x(), go_move.get_y(), go_move.get_stone()==sente.BLACK), move_num))
                game.play(go_move)
            except:
                break
        return transitions, GoImmutableBoard.from_game(game), game.is_over()

    # def finish_game(self, pgn_final_board: ImmutableBoard, move_num: int) -> Tuple[List[Transition], str]:
    #     """Adds transitions to the end of the game."""
    #     lc_zero_transitions = []
    #     board = pgn_final_board.to_board()
    #     while len(lc_zero_transitions) < MAX_GAME_LENGTH:
    #         if self.do_sample_finish:
    #             move = self.lc_zero_policy.sample_move(ImmutableBoard.from_board(board)) #         else:
    #             move = self.lc_zero_policy.get_best_moves(ImmutableBoard.from_board(board), 1)[0]
    #         lc_zero_transitions.append(
    #             Transition(ImmutableBoard.from_board(board), move, move_num + len(lc_zero_transitions))
    #         )
    #         board.push(move)
    #         if board.is_game_over():
    #             lc_zero_transitions.append(
    #                 Transition(ImmutableBoard.from_board(board), None, move_num + len(lc_zero_transitions))
    #             )
    #             break
    #
    #     return lc_zero_transitions, board.result()

    def create_data(self) -> None:
        part: int = 0

        for n_iterations in range(self.max_games):
            try:
                if(self.aio):
                    game = self.allgames[self.n_games]
                else:
                    self.path_to_sgf_file = self.sgf_files_directories[self.n_games][:-1]
            except:
                print("Run out of games, ending loop")
                break
            train_eval = get_split(n_iterations, self.train_eval_split)
            if not self.only_eval or train_eval == "eval":
                current_dataset = self.select_dataset(train_eval)
                try:
                    game: Optional[GoOneGameData] = self.next_game_to_raw_data()
                    if game is None:
                        break
                    else:
                        self.game_to_datapoints(game, current_dataset)
                        self.n_games += 1
                        self.log_progress(n_iterations)
                except Exception as error:
                    print("Error in game: ", self.path_to_sgf_file, type(error).__name__)
                    print("Tried to open this directory: ", os.path.normpath(os.path.join(self.anchor, self.path_to_sgf_file)))
                    self.n_games += 1
                    self.log_progress(n_iterations)

            if (
                (self.save_path_to_eval_set and self.save_path_to_train_set) is not None
                and n_iterations % self.save_data_every_n_games == 0
                and n_iterations > 0
            ):
                self.save_data(part)
                part += 1

            if (n_iterations%100 == 0):
                print(n_iterations)

        if (
                (self.save_path_to_eval_set and self.save_path_to_train_set) is not None
        ):
            self.save_data(part)
            part += 1


        self.data_constructed = True

    def get_train_set_generator(self) -> GoDataset:
        assert self.data_constructed, "Data not constructed, call .create_data() first"
        return GoDataset(self.train_data_queue)

    def get_eval_set_generator(self) -> GoDataset:
        assert self.data_constructed, "Data not constructed, call .create_data() first"
        return GoDataset(self.eval_data_queue)

    def save_data(self, part: int) -> None:
        pd_tarin: pd.DataFrame = pd.DataFrame(self.train_data_queue).transpose()
        pd_eval: pd.DataFrame = pd.DataFrame(self.eval_data_queue).transpose()
        print("Saving Train data: ", pd_tarin)
        print("To path: ", self.save_path_to_train_set + f"{self.name_of_sgf_file}_train_part_{part}.pkl")
        pd_tarin.to_pickle(self.save_path_to_train_set + f"{self.name_of_sgf_file}_train_part_{part}.pkl")
        pd_eval.to_pickle(self.save_path_to_eval_set + f"{self.name_of_sgf_file}_eval_part_{part}.pkl")

        self.train_data_queue.clear()
        self.eval_data_queue.clear()

    def select_dataset(self, train_eval: str) -> Dict:
        if train_eval == "train":
            current_dataset = self.train_data_queue
        elif train_eval == "eval":
            current_dataset = self.eval_data_queue
        else:
            raise ValueError(f"Unknown train_eval value. Expected 'train' or 'eval', got {train_eval}")
        return current_dataset

    def game_to_datapoints(self, one_game_data: GoOneGameData, current_dataset: Dict) -> None:
        """Converts a game to datapoints added to the current_dataset (train or eval)."""
        raise NotImplementedError

    def log_sample(self, sample: Any, game_metadata: GoMetadata) -> None:
        if self.log_samples_limit is not None and random.random() <= self.p_log_sample:
            if self.logged_samples < self.log_samples_limit:
                #log_object("Data sample", self.sample_to_log_object(sample, game_metadata)) ?????????????????????
                self.logged_samples += 1

    def sample_to_log_object(self, sample: Any, game_metadata: GoMetadata) -> Any:
        raise NotImplementedError

    def log_progress(self, n_iterations: int) -> None:
        if n_iterations % self.log_stats_after_n_games == 0:
            log_value(
                f"Train dataset points in batch {self.save_data_every_n_games}",
                n_iterations,
                len(self.train_data_queue),
            )
            log_value(
                f"Eval dataset points in batch {self.save_data_every_n_games}", n_iterations, len(self.eval_data_queue)
            )
            log_value("Dataset games", n_iterations, self.n_games)
            if n_iterations % self.save_data_every_n_games != 0:
                log_value(
                    "Dataset size",
                    n_iterations,
                    self.size_of_computed_dataset + len(self.eval_data_queue) + len(self.train_data_queue),
                )
            else:
                self.size_of_computed_dataset += len(self.eval_data_queue) + len(self.train_data_queue)
                log_value("Dataset size", n_iterations, self.size_of_computed_dataset)


class SimpleGamesDataGenerator(GoGamesDataGenerator):
    '''
    Function generates data suitable for training convolutional networks for Go. It returns a dataframe, with the current state of board
    (19x19x5 - Black/White/Empty/Ko/Colour of the next player to move) 
    in the first column, and the next move in the game (x: 0-18, y: 0-18, Black:True/False) 
    in the second column
    '''
    def game_to_datapoints(self, one_game_data: GoOneGameData, current_dataset: Dict):
        for num, transition in enumerate(one_game_data.transitions):
            if random.random() <= self.p_sample and self.go_filter.use_transition(transition, one_game_data):
                board = transition.immutable_board.boards
                move = transition.move
                current_dataset[len(current_dataset)] = {
                    "input_ids": np.append(board, np.full((board.shape[0], board.shape[1], 1) 
                                            , transition.move[-1]), axis = -1),
                    "labels": (move[0], move[1]) 
                }
                sample = {"input_board": transition.immutable_board, "move": transition.move, "num": num}
                self.log_sample(sample, one_game_data.metadata)

    def sample_to_log_object(self, sample: Any, metadata: GoMetadata):
        # Not implemented yet - this is to log some samples for debugging - in chess it was creating example board states as pictures
        return 0

class SimpleGamesDataGeneratorWithHistory(GoGamesDataGenerator):
    '''
    Function generates data suitable for training convolutional networks for Go. It returns a dataframe, 
    with the current state of board
    (19x19x17 - Black/White binary feature for 8 subsequent moves, Colour of the next player to move) 
    in the first column, and the next move in the game (x: 0-18, y: 0-18, Black:True/False) 
    in the second column
    '''
    def game_to_datapoints(self, one_game_data: GoOneGameData, current_dataset: Dict):
        
        num_transitions = len(one_game_data.transitions)
        reversed_transitions = list(reversed(one_game_data.transitions))
        for num, transition in enumerate(reversed_transitions):
            if random.random() <= self.p_sample and self.go_filter.use_transition(transition, one_game_data):

                #If more than 8 moves left concatenate them else pad with zeros
                if num + 7 < num_transitions: 
                    board = np.concatenate([reversed_transitions[num + i].immutable_board.boards[:,:,:2] \
                                            for i in range(8)], axis = -1)
                else:
                    num_pad = 8 - (num_transitions - num) # Pad moves before first move
                    board = np.concatenate([reversed_transitions[num + i].immutable_board.boards[:, :, :2] 
                                            for i in range(num_transitions - num)] + 
                                            [np.zeros((19,19, num_pad*2))], axis = -1)

                move = transition.move
                current_dataset[len(current_dataset)] = {
                    "input_ids": np.concatenate((board, np.full((board.shape[0], board.shape[1], 1) 
                                            , transition.move[-1])), axis = -1),
                    "labels": (move[0], move[1]) 
                }
                sample = {"input_board": transition.immutable_board, "move": transition.move, "num": num}
                self.log_sample(sample, one_game_data.metadata)

    def sample_to_log_object(self, sample: Any, metadata: GoMetadata):
        # Not implemented yet - this is to log some samples for debugging - in chess it was creating example board states as pictures
        return 0

class GoSimpleGamesDataGeneratorTokenized(GoGamesDataGenerator):
    def game_to_datapoints(self, one_game_data: GoOneGameData, current_dataset: Dict):
        for num, transition in enumerate(one_game_data.transitions):
            if random.random() <= self.p_sample and self.go_filter.use_transition(transition, one_game_data):
                current_dataset[len(current_dataset)] = {
                    "input_ids": GoTokenizer.encode_immutable_board(transition.immutable_board)
                    #+ [ChessTokenizer.vocab_to_tokens["<SEP>"]]
                    ,
                    "labels": GoTokenizer.encode_move(transition.move),
                }
                sample = {"input_board": transition.immutable_board, "move": transition.move, "num": num}
                self.log_sample(sample, one_game_data.metadata)

    def sample_to_log_object(self, sample: Any, metadata: GoMetadata):
        # Not implemented yet - this is to log some samples for debugging - in chess it was creating example board states as pictures
        # return immutable_boards_to_img(
        #     [sample["input_board"]],
        #     [f"{sample['num']} : {sample['move']}, result: {metadata.Result}"],
        # )
        return 0

class GoSimpleGamesDataGeneratorTokenizedAlwaysBlack(GoGamesDataGenerator):
    def game_to_datapoints(self, one_game_data: GoOneGameData, current_dataset: Dict):
        for num, transition in enumerate(one_game_data.transitions):
            if random.random() <= self.p_sample and self.go_filter.use_transition(transition, one_game_data):
                curr_move = transition.move
                curr_boards = transition.immutable_board.boards

                if(curr_move[2]==False):
                    fin_boards = curr_boards[:,:,[1,0,2,3]]
                else:
                    fin_boards = curr_boards
                fin_move = (curr_move[0], curr_move[1], True)

                current_dataset[len(current_dataset)] = {
                    "input_ids": GoTokenizer.encode_boards(fin_boards, True)
                    #+ [ChessTokenizer.vocab_to_tokens["<SEP>"]]
                    ,
                    "labels": GoTokenizer.encode_move(fin_move),
                }
                sample = {"input_board": transition.immutable_board, "move": transition.move, "num": num}
                self.log_sample(sample, one_game_data.metadata)

    def sample_to_log_object(self, sample: Any, metadata: GoMetadata):
        # Not implemented yet - this is to log some samples for debugging - in chess it was creating example board states as pictures
        # return immutable_boards_to_img(
        #     [sample["input_board"]],
        #     [f"{sample['num']} : {sample['move']}, result: {metadata.Result}"],
        # )
        return 0


class GoValueTokenized(GoGamesDataGenerator):
    '''3 - White win; 4 - Black win'''
    def game_to_datapoints(self, one_game_data: GoOneGameData, current_dataset: Dict):
        if one_game_data.metadata.RE[0] == 'B':
            result = [4]
        elif one_game_data.metadata.RE[0] == 'W':
            result = [3]
        else:
            print("Score not clear: ", one_game_data.metadata.RE)
            return

        for num, transition in enumerate(one_game_data.transitions):
            if random.random() <= self.p_sample and self.go_filter.use_transition(transition, one_game_data):
                #print(f"Metadata: {[one_game_data.metadata[k] for k in one_game_data.metadata] }")

                #print(f"Saving as result: {result}")
                current_dataset[len(current_dataset)] = {
                    "input_ids": GoTokenizer.encode_immutable_board(transition.immutable_board)
                    #+ [ChessTokenizer.vocab_to_tokens["<SEP>"]]
                    ,
                    "labels": result,
                }
                sample = {"input_board": transition.immutable_board, "move": transition.move, "num": num}
                self.log_sample(sample, one_game_data.metadata)

    def sample_to_log_object(self, sample: Any, metadata: GoMetadata):
        # Not implemented yet - this is to log some samples for debugging - in chess it was creating example board states as pictures
        # return immutable_boards_to_img(
        #     [sample["input_board"]],
        #     [f"{sample['num']} : {sample['move']}, result: {metadata.Result}"],
        # )
        return 0

import copy
class GoSubgoalGamesDataGenerator(GoGamesDataGenerator):
    def __init__(
            self,
            number_of_datapoint_from_one_game: int = 1,
            range_of_k: List[int] = [3],
            take_all_datapoints: bool = True,
            *args,
            **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.range_of_k = range_of_k
        self.number_of_datapoint_from_one_game = number_of_datapoint_from_one_game
        #self.prob_selector: Dict[int, np.ndarray] = prob_table_for_diff_n((1, 800))
        self.take_all_datapoints = take_all_datapoints

        if self.take_all_datapoints:
            print("Will be taken all datapoints from one game")
            print("Number of datapoints from one game will be ignored")

        # assert (
        #         self.range_of_k is not None and self.number_of_datapoint_from_one_game is not None
        # ), "Set range_of_k and number_of_datapoint_from_one_game must be set"

        self.save_path_to_train_set: str = os.path.join(self.save_path_to_train_set)
        self.save_path_to_eval_set: str = os.path.join(self.save_path_to_eval_set)

        try:
            os.makedirs(self.save_path_to_train_set)
            os.makedirs(self.save_path_to_eval_set)
            print(f"Directory {self.save_path_to_train_set} created successfully")
            print(f"Directory {self.save_path_to_eval_set} created successfully")
        except OSError as error:
            print(
                f"Directory {self.save_path_to_eval_set} or {self.save_path_to_train_set} can not be created. "
                f"Probably exists."
                f"Error {error}"
            )

    def game_to_datapoints(
            self, one_game_data: GoOneGameData, current_dataset: Dict[int, Dict[str, Union[List[int], str, int]]]
    ) -> None:
        game_length: int = len(one_game_data.transitions)
        if game_length > 0:
            selected_input_datapoints_all_k: Dict[int, List[int]] = {}
            temporary_dict_datapoints_for_k: Dict[str, Union[List[int], str, int]] = {}
            #pdf: np.ndarray = self.prob_selector[game_length]

            for k in self.range_of_k:
                if self.take_all_datapoints:
                    selected_datapoints = np.array([position for position in range(game_length-self.range_of_k[0])])
                # else:
                #     selected_datapoints = np.random.choice(
                #         list(range(game_length)), size=self.number_of_datapoint_from_one_game, p=pdf, replace=False
                #     )
                selected_input_datapoints_all_k[k] = selected_datapoints

            for games_positions in zip(*selected_input_datapoints_all_k.values()):
                for k, position in zip(self.range_of_k, games_positions):
                    input_board: GoImmutableBoard = one_game_data.transitions[position].immutable_board
                    target_board_num = min(game_length - 1, position + k)
                    target_board: GoImmutableBoard = one_game_data.transitions[target_board_num].immutable_board

                    temporary_dict_datapoints_for_k.update(
                        {
                            # f"input_ids_{k}": GoTokenizer.encode_immutable_board(input_board)
                            #                   + [ChessTokenizer.vocab_to_tokens["<SEP>"]],
                            # f"labels_{k}": GoTokenizer.encode_immutable_board(target_board),
                            f"input_ids": GoTokenizer.encode_immutable_board(input_board),
                            f"labels": GoTokenizer.encode_immutable_board(target_board),
                        }
                    )
                    # temporary_dict_datapoints_for_k.update(
                    #     {
                    #         f"all_moves_from_start_{k}": [
                    #             ChessTokenizer.encode_move(one_game_data.transitions[i].move)[0]
                    #             for i in range(position)
                    #         ]
                    #     }
                    # )
                    # temporary_dict_datapoints_for_k.update(
                    #     {
                    #         f"moves_between_input_and_target_{k}": [
                    #             ChessTokenizer.encode_move(one_game_data.transitions[i].move)[0]
                    #             for i in range(position, target_board_num)
                    #         ]
                    #     }
                    # )
                    # temporary_dict_datapoints_for_k.update({f"move_number_form_start_{k}": position})

                # boards_stats: Dict[str, str] = {"Result": "", "WhiteElo": "", "BlackElo": "", "Opening": ""}
                #
                # for stat in one_game_data.metadata.__dict__:
                #     if stat in boards_stats:
                #         boards_stats[stat] = one_game_data.metadata.__dict__[stat]
                #
                # temporary_dict_datapoints_for_k.update(boards_stats)
                # temporary_dict_datapoints_for_k.update({"game_length": game_length})

                current_dataset[len(current_dataset)] = copy.deepcopy(temporary_dict_datapoints_for_k)
                temporary_dict_datapoints_for_k.clear()

    def sample_to_log_object(self, sample: Dict, metadata: GoMetadata):
        return 0
        # return immutable_boards_to_img(
        #     [sample["input_board"], sample["target_board"]],
        #     [f"{sample['num']} : {sample['move']}, res: {metadata.Result}", ""],
        # )




import re

if __name__ == '__main__':

    # generator = SimpleGamesDataGenerator(sgf_files='sgf_directories.txt',save_data_every_n_games=1,p_sample=1,max_games=2,train_eval_split=1,save_path_to_eval_set='eval',save_path_to_train_set='train')
    # generator.create_data()

    # generator = GoSimpleGamesDataGeneratorTokenizedAlwaysBlack(sgf_files='sgf_directories.txt',save_data_every_n_games=1,p_sample=1,max_games=2,train_eval_split=1,save_path_to_eval_set='tokenized_data\\eval',save_path_to_train_set='tokenized_data\\train')
    # generator.create_data()

    # generator = GoSimpleGamesDataGeneratorTokenizedAlwaysBlack(sgf_files='val.txt',save_data_every_n_games=101,p_sample=1,max_games=103,train_eval_split=0.95,save_path_to_eval_set='tokenized_data\\eval',save_path_to_train_set='tokenized_data\\train')
    # generator.create_data()

    # generator = GoValueTokenized(sgf_files='sgf_directories.txt',save_data_every_n_games=119,p_sample=1,max_games=103,train_eval_split=0.95,save_path_to_eval_set='tokenized_data/value/test/eval/',save_path_to_train_set='tokenized_data/value/test/train/')
    # generator.create_data()

    # generator = GoSubgoalGamesDataGenerator(sgf_files='2005-11.aio',save_data_every_n_games=119,p_sample=1,max_games=100,train_eval_split=0.95,save_path_to_eval_set='tokenized_data/subgoal/test/eval/',save_path_to_train_set='tokenized_data/subgoal/test/train/')
    # generator = GoSubgoalGamesDataGenerator(sgf_files='sgf_directories.txt', save_data_every_n_games=119,
    #                                         p_sample=1, max_games=10, train_eval_split=0.95,
    #                                         save_path_to_eval_set='tokenized_data/subgoal/test/eval/',
    #                                         save_path_to_train_set='tokenized_data/subgoal/test/train/')

    # generator.create_data()

    #print(os.path.basename('C:\\Users\\Antek\\PycharmProjects\\subgoal_search_chess\\data_processing\\val.txt')[:-4])

    np.set_printoptions(threshold=10000)
    aaa = pd.read_pickle("/mnt/c/Users/Antek/PycharmProjects/subgoal_search_chess/data_processing/tokenized_data/subgoal/test/train/2005-11_train_part_0.pkl")
    print(aaa.iloc[0]['labels'])


    # #print(os.path.join(dirs, sgf))

    # #print(os.path.dirname(dirs))

    # for a in os.walk("."):
    #    print(a)


    # game = sente.sgf.loads('(;BR[]WR[]KM[]RE[];B[qd];W[dp];B[ec];W[pp];B[cn];W[cl];B[df];W[dn];B[do];W[co];B[eo];W[en];B[cp];W[bo];B[fn];W[fm];B[fo];W[cq];B[em];W[dm];B[el];W[gm];B[ho];W[cj];B[ej];W[im];B[jo];W[km];B[lo];W[cd];B[ee];W[hj];B[ci];W[bi];B[dj];W[bk];B[ch];W[bf];B[kd];W[mq];B[no];W[op];B[pn];W[pl];B[mm];W[pi];B[qg];W[oc];B[mk];W[kj];B[li];W[pf];B[qf];W[qc];B[rc];W[pd];B[qe];W[oh];B[ki];W[ji];B[jh];W[ii];B[kk];W[hg];B[pe];W[oe];B[og];W[on];B[pm];W[ql];B[om];W[md];B[of];W[qh];B[qb];W[kb];B[ic];W[jg];B[kg];W[ih];B[jf];W[kh];B[lh];W[eh];B[dh];W[kc];B[ld];W[lc];B[jh];W[jq];B[ig];W[jk];B[lj];W[ib];B[hb];W[hc];B[jj];W[ik];B[hd];W[gc];B[gk];W[gd];B[ge];W[hl];B[hn];W[gh];B[jc];W[jb];B[hf];W[ei];B[di];W[eg];B[cf];W[fb];B[rm];W[rl];B[qq];W[qp];B[rp];W[pq];B[rr];W[qr];B[rq];W[bg];B[gg];W[hh];B[jn];W[ff];B[bh];W[ah];B[gl];W[hm];B[gf];W[jm];B[fg];W[fh];B[kl];W[jl];B[ce];W[be];B[fd];W[eb];B[lq];W[lr];B[mr];W[nr];B[kr];W[ms];B[kq];W[fq];B[np];W[nq];B[pr];W[or];B[qs];W[os];B[dc];W[cc];B[eq];W[dq];B[fr];W[er];B[gq];W[ep];B[fp];W[ef];B[fe];W[dd];B[de];W[fc];B[ed];W[db];B[eq];W[ks];B[js];W[ls];B[ok];W[rj];B[jr];W[fq];B[ob];W[nb];B[pc];W[od];B[na];W[ma];B[oa];W[mc];B[rh];W[ri];B[rg];W[qm];B[rn];W[qn];B[qo];W[po];B[ro];W[ol];B[nm];W[oj];B[nk];W[pk];B[eq];W[le];B[dr];W[cr];B[fj];W[gj];B[es];W[jd];B[id];W[ke];B[je];W[cs];B[bn];W[bp];B[dl];W[cm];B[oo];W[oq])(;')
    # game.play_default_sequence()
    # print(game)