import os
from typing import Dict, Any, Optional, List, Type, Union, Tuple

import chess.pgn
import random
import numpy as np

import pandas as pd
import torch
from collections import namedtuple

from matplotlib import pyplot as plt
from tqdm import tqdm

from configures.global_config import MAX_GAME_LENGTH
from data_processing.chess_tokenizer import ChessTokenizer
from data_structures.data_structures import ImmutableBoard, ChessMetadata, Transition, OneGameData
from data_processing.data_utils import get_split, immutable_boards_to_img, RESULT_TO_WINNER
from metric_logging import log_value, log_object

from data_processing.probability_subgoal_selector_tools import prob_table_for_diff_n, prob_select_function
from policy.chess_policy import LCZeroPolicy

# TODO: fill in fields
GameMetadata = namedtuple("GameMetadata", "game_id, winner, result")


class ChessFilter:
    """Filters games and transitions to use in training."""

    def use_game(self, game_metadata: ChessMetadata) -> bool:
        """Decides whether a game is useful."""
        raise NotImplementedError

    def use_transition(self, transition: Transition, one_game_data: OneGameData) -> bool:
        """Decides whether a transition is useful."""
        raise NotImplementedError


class NoFilter(ChessFilter):
    """Accepts every game and transition"""

    def use_game(self, game_metadata: ChessMetadata) -> bool:
        return True

    def use_transition(self, transition: Transition, one_game_data: OneGameData) -> bool:
        return True


class ResultFilter(ChessFilter):
    """Filters games that have a winner or looser. Filters transitions that were played by the winner or looser."""

    def __init__(
        self,
        winner_or_looser: str = "winner",
    ) -> None:
        assert winner_or_looser in ["winner", "loser"], "winner_or_looser must be 'winner' or 'loser'"
        self.winner_or_looser = winner_or_looser

    def use_game(self, game_metadata: ChessMetadata) -> bool:
        take_game: bool = False
        try:
            take_game = game_metadata.Result in ["1-0", "0-1"]
        except Exception as e:
            print(f"Error: {e}")
            print("\n")
            print("Can't find inforations about game result. Return False.")
        return take_game

    def use_transition(self, transition: Transition, one_game_data: OneGameData) -> bool:

        if self.winner_or_looser == "winner":
            return transition.immutable_board.active_player == RESULT_TO_WINNER[one_game_data.metadata.Result]
        elif self.winner_or_looser == "loser":
            return transition.immutable_board.active_player != RESULT_TO_WINNER[one_game_data.metadata.Result]


class ELOFilter(ChessFilter):
    def __init__(self, elo_threshold: int = 1000):
        self.elo_threshold = elo_threshold

    def use_game(self, game_metadata: ChessMetadata) -> bool:
        take_game: bool = False
        try:
            take_game = min(int(game_metadata.WhiteElo), int(game_metadata.BlackElo)) >= self.elo_threshold
        except Exception as e:
            print(f"Error: {e}")
            print("\n")
            print("Can't find inforations about ELO. Return False.")
        return take_game

    def use_transition(self, transition: Transition, one_game_data: OneGameData) -> bool:
        """Decides whether a transition is useful."""
        raise NotImplementedError


class ChessDataset(torch.utils.data.Dataset):
    """Used by Pytorch DataLoader to get batches of data."""

    def __init__(self, data: Dict):
        self.data = data

    def __getitem__(self, idx: int):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


class ChessDataProvider:
    """General class for providing data for training and evaluation."""

    def get_train_set_generator(self) -> ChessDataset:
        raise NotImplementedError

    def get_eval_set_generator(self) -> ChessDataset:
        raise NotImplementedError


class ChessGamesDataGenerator(ChessDataProvider):
    """Reads PGN file and creates data."""

    def __init__(
        self,
        pgn_file: Optional[str] = None,
        chess_filter: Optional[Type[ChessFilter]] = None,
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
        self.path_to_pgn_file = pgn_file
        self.name_of_pgn_file: str = self.path_to_pgn_file.split("/")[-1]
        self.pgn_database = open(self.path_to_pgn_file, errors="ignore")
        assert chess_filter is not None, "Chess filter must be specified"
        self.chess_filter = chess_filter()

        print(100 * "=")
        print(
            f"Name of used filter: {type(self.chess_filter)}." f" Atributs of used filter: {self.chess_filter.__dict__}"
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

        self.lc_zero_policy = LCZeroPolicy()

    def next_game_to_raw_data(self) -> Optional[OneGameData]:
        """
        Function takes the next game from the set of chess games, checks if it passes through the filter used and
        return information about game chass.
        :return: OneGameData contains inforamtion about chess game.
        """

        current_game: chess.pgn.Game = chess.pgn.read_game(self.pgn_database)

        if current_game is None:  # Condition is met if there are no more games in the dataset.
            return None

        chess_metadata: ChessMetadata = ChessMetadata(**current_game.headers)
        transitions: List[Transition] = []

        if self.chess_filter.use_game(chess_metadata):
            transitions, final_pgn_board, is_game_over = self.moves_to_transitions(
                initial_immutable_board=None, moves=current_game.mainline_moves()
            )

            if not is_game_over:
                lc_zero_transitions, result = self.finish_game(
                    pgn_final_board=final_pgn_board, move_num=len(transitions)
                )

                transitions.extend(lc_zero_transitions)
                chess_metadata.Result = result

        # if self.save_filtered_data is not None:
        #     completed_game = chess.pgn.Game()
        #     completed_game.headers = chess_metadata.__dict__
        #     completed_game.setup(final_pgn_board)
        #
        #     print(f"Filtered data will be saved in: {self.save_filtered_data}")
        #     if isinstance(self.chess_filter, ELOFilter):
        #         self.save_filtered_data = os.path.join(self.save_filtered_data,
        #                                                f"{type(self.chess_filter)}_ELO_{self.chess_filter.elo_threshold}.pgn")
        #     else:
        #         self.save_filtered_data = os.path.join(self.save_filtered_data, f"_{type(self.chess_filter)}.pgn")
        #
        #     with open(self.save_filtered_data, "a") as f:
        #         print(current_game, file=f)
        #         print("\n", file=f)

        return OneGameData(chess_metadata, transitions)

    def moves_to_transitions(
        self, initial_immutable_board: Optional[ImmutableBoard], moves
    ) -> Tuple[List[Transition], ImmutableBoard, bool]:
        if initial_immutable_board is None:
            board = chess.Board()
        else:
            board = initial_immutable_board.to_board()

        transitions: List[Transition] = []
        for move in enumerate(moves):
            move_num, chess_move = move
            try:
                transitions.append(Transition(ImmutableBoard.from_board(board), chess_move, move_num))
                board.push(chess_move)
            except:
                break
        return transitions, ImmutableBoard.from_board(board), board.is_game_over()

    def finish_game(self, pgn_final_board: ImmutableBoard, move_num: int) -> Tuple[List[Transition], str]:
        """Adds transitions to the end of the game."""
        lc_zero_transitions = []
        board = pgn_final_board.to_board()
        while len(lc_zero_transitions) < MAX_GAME_LENGTH:
            if self.do_sample_finish:
                move = self.lc_zero_policy.sample_move(ImmutableBoard.from_board(board))
            else:
                move = self.lc_zero_policy.get_best_moves(ImmutableBoard.from_board(board), 1)[0]
            lc_zero_transitions.append(
                Transition(ImmutableBoard.from_board(board), move, move_num + len(lc_zero_transitions))
            )
            board.push(move)
            if board.is_game_over():
                lc_zero_transitions.append(
                    Transition(ImmutableBoard.from_board(board), None, move_num + len(lc_zero_transitions))
                )
                break

        return lc_zero_transitions, board.result()

    def create_data(self) -> None:
        part: int = 0

        for n_iterations in range(self.max_games):
            train_eval = get_split(n_iterations, self.train_eval_split)
            if not self.only_eval or train_eval == "eval":
                current_dataset = self.select_dataset(train_eval)
                game: Optional[OneGameData] = self.next_game_to_raw_data()
                if game is None:
                    break
                else:
                    self.game_to_datapoints(game, current_dataset)
                    self.n_games += 1
                    self.log_progress(n_iterations)

            if (
                (self.save_path_to_eval_set and self.save_path_to_train_set) is not None
                and n_iterations % self.save_data_every_n_games == 0
                and n_iterations > 0
            ):
                self.save_data(part)
                part += 1

        self.data_constructed = True

    def get_train_set_generator(self) -> ChessDataset:
        assert self.data_constructed, "Data not constructed, call .create_data() first"
        return ChessDataset(self.train_data_queue)

    def get_eval_set_generator(self) -> ChessDataset:
        assert self.data_constructed, "Data not constructed, call .create_data() first"
        return ChessDataset(self.eval_data_queue)

    def save_data(self, part: int) -> None:
        pd_tarin: pd.DataFrame = pd.DataFrame(self.train_data_queue).transpose()
        pd_eval: pd.DataFrame = pd.DataFrame(self.eval_data_queue).transpose()
        pd_tarin.to_pickle(self.save_path_to_train_set + f"{self.name_of_pgn_file}_train_part_{part}.pkl")
        pd_eval.to_pickle(self.save_path_to_eval_set + f"{self.name_of_pgn_file}_eval_part_{part}.pkl")

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

    def game_to_datapoints(self, one_game_data: OneGameData, current_dataset: Dict) -> None:
        """Converts a game to datapoints added to the current_dataset (train or eval)."""
        raise NotImplementedError

    def log_sample(self, sample: Any, game_metadata: ChessMetadata) -> None:
        if self.log_samples_limit is not None and random.random() <= self.p_log_sample:
            if self.logged_samples < self.log_samples_limit:
                log_object("Data sample", self.sample_to_log_object(sample, game_metadata))
                self.logged_samples += 1

    def sample_to_log_object(self, sample: Any, game_metadata: ChessMetadata) -> Any:
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


class PolicyGamesDataGenerator(ChessGamesDataGenerator):
    def game_to_datapoints(self, one_game_data: OneGameData, current_dataset: Dict):
        for num, transition in enumerate(one_game_data.transitions):
            if random.random() <= self.p_sample and self.chess_filter.use_transition(transition, one_game_data):
                current_dataset[len(current_dataset)] = {
                    "input_ids": ChessTokenizer.encode_immutable_board(transition.immutable_board)
                    + [ChessTokenizer.vocab_to_tokens["<SEP>"]],
                    "labels": ChessTokenizer.encode_move(transition.move),
                }
                sample = {"input_board": transition.immutable_board, "move": transition.move.uci(), "num": num}
                self.log_sample(sample, one_game_data.metadata)

    def sample_to_log_object(self, sample: Any, metadata: ChessMetadata) -> plt.Figure:
        return immutable_boards_to_img(
            [sample["input_board"]],
            [f"{sample['num']} : {sample['move']}, result: {metadata.Result}"],
        )


class ChessSubgoalGamesDataGenerator(ChessGamesDataGenerator):
    def __init__(self, k: int, number_of_datapoint_from_one_game: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_path_to_train_set: str = os.path.join(
            self.save_path_to_train_set, "subgoals_k=" + str(k), "train" + "/"
        )
        self.save_path_to_eval_set: str = os.path.join(self.save_path_to_eval_set, "subgoals_k=" + str(k), "eval" + "/")

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

        self.k = k
        self.prob_selector = prob_table_for_diff_n((5, 500))
        self.number_of_datapoint_from_one_game = number_of_datapoint_from_one_game

    def game_to_datapoints_all(self, one_game_data: OneGameData, current_dataset: List[Dict[str, List[int]]]) -> None:
        game_length = len(one_game_data.transitions)
        for num in range(game_length - self.k):

            input_board = one_game_data.transitions[num].immutable_board
            target_board_num = min(game_length - 1, num + self.k)
            target_board = one_game_data.transitions[target_board_num].immutable_board

            if random.random() <= self.p_sample:
                current_dataset[len(current_dataset)] = {
                    "input_ids": ChessTokenizer.encode_immutable_board(input_board)
                    + [ChessTokenizer.vocab_to_tokens["<SEP>"]],
                    "labels": ChessTokenizer.encode_immutable_board(target_board),
                }

            sample = {
                "input_board": input_board,
                "target_board": target_board,
                "num": num,
                "move": one_game_data.transitions[num].move.uci(),
            }
            self.log_sample(sample, one_game_data.metadata)

    def game_to_datapoints(
        self, one_game_data: OneGameData, current_dataset: List[Dict[str, Union[List[int], str, int]]]
    ) -> None:

        game_length: int = len(one_game_data.transitions)
        if game_length > 0 and game_length - self.k >= 0:

            if game_length <= len(self.prob_selector):
                p: np.ndarray = self.prob_selector[game_length - 1]
            else:
                p: np.ndarray = prob_select_function(game_length - 1)

            assert (
                self.number_of_datapoint_from_one_game is not None
            ), "Please select number of datapoints frome game. Must be integer."

            selected_datapoints = np.random.choice(
                list(range(game_length - 1)), size=self.number_of_datapoint_from_one_game, p=p
            )

            # selected_datapoints = list(range(game_length - 1))

            for key in selected_datapoints:
                input_board: ImmutableBoard = one_game_data.transitions[key].immutable_board
                target_board_num = min(game_length - 1, key + self.k)
                target_board: ImmutableBoard = one_game_data.transitions[target_board_num].immutable_board
                position: int = len(current_dataset)
                current_dataset[position] = {
                    "input_ids": ChessTokenizer.encode_immutable_board(input_board)
                    + [ChessTokenizer.vocab_to_tokens["<SEP>"]],
                    "labels": ChessTokenizer.encode_immutable_board(target_board),
                }
                all_moves_from_start: Dict[str, List[int]] = {
                    "all_moves_from_start": [
                        ChessTokenizer.encode_move(one_game_data.transitions[i].move)[0] for i in range(key)
                    ]
                }
                moves_between_input_and_target: Dict[str, List[int]] = {
                    "moves_between_input_and_target": [
                        ChessTokenizer.encode_move(one_game_data.transitions[i].move)[0]
                        for i in range(key, target_board_num)
                    ]
                }

                boards_stats: Dict[str, str] = {"Result": "", "WhiteElo": "", "BlackElo": "", "Opening": ""}

                for stat in one_game_data.metadata.__dict__:
                    if stat in boards_stats:
                        boards_stats[stat] = one_game_data.metadata.__dict__[stat]

                current_dataset[position].update(all_moves_from_start)
                current_dataset[position].update(moves_between_input_and_target)
                current_dataset[position].update(boards_stats)
                current_dataset[position].update({"move_number_form_start": key})
                current_dataset[position].update({"game_length": game_length})

    def sample_to_log_object(self, sample: Dict, metadata: ChessMetadata) -> plt.Figure:
        return immutable_boards_to_img(
            [sample["input_board"], sample["target_board"]],
            [f"{sample['num']} : {sample['move']}, res: {metadata.Result}", ""],
        )


class ChessCLLPGamesDataGenerator(ChessGamesDataGenerator):
    def __init__(self, max_k, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_k = max_k

    def game_to_datapoints(self, one_game_data: OneGameData, current_dataset: Dict) -> None:
        game_length = len(one_game_data.transitions)
        for num in range(game_length - 1):

            input_board = one_game_data.transitions[num].immutable_board
            max_target_board_num = min(game_length - 1, num + self.max_k)
            target_board_num = random.randint(num + 1, max_target_board_num)
            move = one_game_data.transitions[num].move
            all_cllp_moves = [one_game_data.transitions[i].move for i in range(num + 1, target_board_num + 1)]
            target_board = one_game_data.transitions[target_board_num].immutable_board

            if random.random() <= self.p_sample:
                current_dataset[len(current_dataset)] = {
                    "input_ids": ChessTokenizer.encode_immutable_board(input_board)
                    + ChessTokenizer.encode_immutable_board(input_board)
                    + [ChessTokenizer.vocab_to_tokens["<SEP>"]],
                    "labels": ChessTokenizer.encode_move(move),
                }

            sample = {
                "input_board": input_board,
                "target_board": target_board,
                "num": num,
                "move": move.uci(),
                "all_cllp_moves": all_cllp_moves,
            }
            self.log_sample(sample, one_game_data.metadata)

    def sample_to_log_object(self, sample: Dict, metadata: ChessMetadata) -> plt.Figure:
        return immutable_boards_to_img(
            [sample["input_board"], sample["target_board"]],
            [f"{sample['num']} : {sample['all_cllp_moves']}, res: {metadata.Result}", ""],
        )
