import chess
import chess.pgn
import chess.engine

import pandas as pd
import matplotlib.pyplot as plt

from typing import Optional, List, Dict, Tuple, Any
import random
from tqdm import tqdm

from data_processing.data_utils import immutable_boards_to_img, get_split
from data_structures.data_structures import ImmutableBoard, OneGameData, ChessMetadata, Transition
from subgoal_generator.subgoal_generator import BasicChessSubgoalGenerator
from chess_engines.stockfish import evaluate_immutable_board_by_stockfish_with_resret_machine
from data_processing.chess_data_generator import NoFilter, ChessFilter, ChessDataGenerator

import chess.svg


def n_forwad_moves_chess_board(
        moves_during_play: List[Tuple[str, chess.Move]],
        number_of_forward_moves: int
) -> Tuple[chess.Board, str]:
    """
    For example: Moves during playgame:  1. e4 c5 2. f4 d6 3. Nf3 Nc6 4. Bb5 Bd7 5. c3 a6 6.
    Number of forward moves: 1.

    r n b q k b n r         r n b q k b n r
    p p p p p p p p         p p p p p p p p
    . . . . . . . .         . . . . . . . .
    . . . . . . . .         . . . . . . . .
    . . . . . . . . ------> . . . . . . . .
    . . . . . . . .         . . . . P . . .
    P P P P P P P P         P P P P . P P P
    R N B Q K B N R         R N B Q K B N R

    :param: moves_during_play: These are all moves during current playgame.
    :param: number_of_forward_moves: The number of moves that players have made.
    :return: Board after number_of_forward_moves.
    """

    active_player: Optional[str] = None
    beggining_board: chess.Board = chess.Board()
    number_of_forward_moves: int = min(number_of_forward_moves, len(moves_during_play))

    for i in range(number_of_forward_moves):
        beggining_board.push(moves_during_play[i][1])
        active_player = moves_during_play[i][0]

    return beggining_board, active_player


class ChessStatsDatasetCreator(ChessDataGenerator):
    def __init__(self, pgn_file: str, n_games: int, train_eval_split: float, chess_filter: ChessFilter = NoFilter):
        super().__init__(pgn_file, chess_filter, train_eval_split)

        self.pgn_file = pgn_file
        self.pgn_database = open(pgn_file, errors="ignore")
        self.n_games = n_games  # number of games that we want to evaluate (we want statistics from them)
        self.chess_filter = chess_filter
        self.train_eval_split = train_eval_split

        # list contains chess games that goes (boards from move's trajectory)
        # into the evaluation dataset
        self.dataset_containig_games_to_eval_statistics: Dict[int, Transition] = {}

    def create_data(self) -> None:
        """
        Function splits chess dataset on training and evaluation set (using filtr - by default there is no filter).
        Spliting is the same as in ChessDataGenerator (by get_split function). Only evaluation set is saved.
        :return: None
        """
        n_iterations: int = 0
        dict_position: int = 0
        n_game_to_eval: int = self.n_games

        while self.n_games > 0:
            n_iterations += 1
            train_eval: str = get_split(n_iterations, self.train_eval_split)
            game = self.next_game_to_raw_data()

            if game is None:  # The entire dataset was used.
                break

            if train_eval == "eval":
                if len(game.transitions) == 0:  # Empty games created after applying the filter.
                    continue
                else:
                    self.dataset_containig_games_to_eval_statistics[dict_position] = game
                    dict_position += 1
                    self.n_games -= 1

        assert len(self.dataset_containig_games_to_eval_statistics) != 0, "No data for evaluation, probably set too " \
                                                                          "large train_eval_split."
        if self.n_games != 0:
            print('The number of games we want to '
                  'evaluate is {0} but {1} are required.'.format(n_game_to_eval - self.n_games, n_game_to_eval))
            print('If more games are needed for evluation, then increase the dataset,'
                  ' change the filter or minimize large train_eval_split parameter.')

    def game_to_datapoints(self, one_game_data: OneGameData, current_dataset: Dict):
        pass

    def sample_to_log_object(self, sample: Any, game_metadata: ChessMetadata) -> Any:
        pass

    def chess_dataset_stats(self) -> pd.DataFrame:

        """
        A functions that counts the number of games, the number of games that white opponent has won,
        the number of games that black opponent has won, the number of games ended in a draw and the and
        the number of games filtred by the filter used.

        For exaple:

        Number of games:   Number of games won by white player:   Number of games won by black player:   Nuber of draws:   Nuber of filtred gamse:
      0                 33                                     15                                      6                12                        33

        :return: Pandas frame which contains dataset statistic information.
        """

        number_of_games: int = 0
        white_won: int = 0
        black_won: int = 0
        draws: int = 0
        filtred_games: int = 0
        database_statistic: dict = {}
        database_of_chess_games_file = open(self.pgn_file)

        while True:
            game = chess.pgn.read_game(database_of_chess_games_file)
            if game is None:  # The entire dataset was used.
                break
            else:
                chess_metadata: ChessMetadata = ChessMetadata(**game.headers)
                game_result = chess_metadata.__dict__['Result']
                number_of_games += 1

                if self.chess_filter.use_game(chess_metadata):
                    filtred_games += 1

                if game_result == '1-0':
                    white_won += 1
                elif game_result == '0-1':
                    black_won += 1
                else:
                    draws += 1

        assert number_of_games == white_won + black_won + draws, 'number_of_games = white_won + black_won + draws'

        database_statistic['Number of games: '] = [number_of_games]
        database_statistic['Number of games won by white player: '] = [white_won]
        database_statistic['Number of games won by black player: '] = [black_won]
        database_statistic['Nuber of draws: '] = [draws]
        database_statistic['Nuber of filtred gamse: '] = [filtred_games]

        df_chess_games_dataset_stats = pd.DataFrame(database_statistic)
        database_of_chess_games_file.close()

        return df_chess_games_dataset_stats


class StatisticOfSubgoals:
    def __init__(
            self,
            chess_stats_eval_dataset_creator: ChessStatsDatasetCreator,
            chess_subgaols_generator_checkpoint_path: str
    ) -> None:

        self.chess_stats_eval_dataset_creator = chess_stats_eval_dataset_creator
        self.chess_subgaols_generator_checkpoint_path = chess_subgaols_generator_checkpoint_path
        self.subgoal_model: BasicChessSubgoalGenerator = \
            BasicChessSubgoalGenerator(self.chess_subgaols_generator_checkpoint_path)

    def diff_value_input_state_vs_subgolas_multi_games(
            self,
            number_of_subgoals: int,
            path_to_folder_to_save_graphics: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Function takes n games form evaluation set, randomly chooses state (from each game)
        and generates subgoals from it. Then evaluates each state and subgoal using stockfish.

        For example:

           Input board evaluation  Subgoal_1 evaluation    active_player   Result  Numbers of forward moves
0                      17                    22               b            1/2-1/2                10
1                      71                    71               b              0-1                  32
2                    -179                  1176               b            1/2-1/2                56

        :param: number_of_subgoals: Number of subgoals we want to generate for current state.
        :param: path_to_folder_to_save_graphics: If None then each line is stored in graphical form.
        :return: Dataframe which consits of evaluations (by stockfish) entire game with subgoals
        and inforamtion about game. Each row represents current evaluated state wwith subgoals form it.
        """

        database_of_chess_games_to_eval: Dict[int, Transition] = \
            self.chess_stats_eval_dataset_creator.dataset_containig_games_to_eval_statistics

        data_stockfish_estimation_state: Dict[str, List] = {'Input board evaluation': []}
        for i in range(1, number_of_subgoals + 1):
            data_stockfish_estimation_state['Subgoal_' + str(i) + ' evaluation'] = []

        # Here you can add information about the games played, which are included in the ChessMetadata or ImmutableBoard

        # Example for ChessMetadata:

        # {'Event': '27th BIH Teams A 2020', 'Site': 'Lukavac BIH', 'Date': '2020.11.04',
        # 'Round': '9.1', 'White': 'Redzepi, Mehmedalija', 'Black': 'Kurtcehajic, Suad', 'Result': '1/2-1/2',
        # 'BlackElo': '1951', 'BlackFideId': '14409585', 'BlackTeam': 'SK Sarajevo, Sarajevo', 'ECO':
        # 'D02', 'EventDate': '2020.10.30', 'EventType': 'team', 'Opening': "Queen's pawn game",
        # 'WhiteElo': '1859', 'WhiteFideId': '14405032', 'WhiteTeam': 'SK Preporod, Zenica'}

        # Example of ImmutableBoard:
        # active_player = 'w', castles = 'KQkq', en_passant_target = '-', halfmove_clock = '0', fullmove_clock = '1'

        data_stockfish_estimation_state['active_player'] = []
        data_stockfish_estimation_state['Result'] = []

        # 'Numbers of forward moves' is not in ChessMetadata or ImmutableBoard information.
        data_stockfish_estimation_state['Numbers of forward moves'] = []

        eval_boards_key_names: List[str] = \
            ['Input board evaluation'] + ['Subgoal_' + str(i) + ' evaluation' for i in range(1, number_of_subgoals + 1)]

        for number_of_current_game in tqdm(range(len(database_of_chess_games_to_eval))):
            current_game: OneGameData = database_of_chess_games_to_eval[number_of_current_game]
            info_about_current_game: Dict[str, str] = current_game[0].__dict__
            moves_in_current_game: List[Tuple[str, chess.Move]] = [(move[0].active_player, move[1]) for move in
                                                                   current_game.transitions]
            number_of_forward_moves: int = random.randrange(0, len(moves_in_current_game))
            board_after_n_moves: Tuple[chess.Board, str] = n_forwad_moves_chess_board(moves_in_current_game,
                                                                                      number_of_forward_moves
                                                                                      )
            immutable_board_after_n_moves: ImmutableBoard = ImmutableBoard.from_board(board_after_n_moves[0])
            subgoals_from_board_after_n_moves: List[ImmutableBoard] = \
                self.subgoal_model.generate_subgoals(immutable_board_after_n_moves, number_of_subgoals)

            # update evaluation by stockfish

            for p, eval_name in enumerate(eval_boards_key_names):
                if eval_name == 'Input board evaluation':
                    board_after_n_moves_eval: float = \
                        evaluate_immutable_board_by_stockfish_with_resret_machine(immutable_board_after_n_moves)
                    data_stockfish_estimation_state[eval_name].append(board_after_n_moves_eval)
                else:
                    subgoals_from_board_after_n_moves_eval: float = \
                        evaluate_immutable_board_by_stockfish_with_resret_machine(
                            subgoals_from_board_after_n_moves[p - 1])
                    data_stockfish_estimation_state[eval_name].append(subgoals_from_board_after_n_moves_eval)

            # update of the rest of the data

            data_stockfish_estimation_state['active_player'].append(board_after_n_moves[1])
            data_stockfish_estimation_state['Result'].append(info_about_current_game['Result'])
            data_stockfish_estimation_state['Numbers of forward moves'].append(number_of_forward_moves)

            if isinstance(path_to_folder_to_save_graphics, str):
                fig: plt.figure = immutable_boards_to_img(
                    [immutable_board_after_n_moves] + subgoals_from_board_after_n_moves,
                    [name + ': ' + str(data_stockfish_estimation_state[name][number_of_current_game])
                     for name in eval_boards_key_names]
                )
                fig.suptitle('Active player: {0}, result: {1}'.format(
                    data_stockfish_estimation_state['active_player'][number_of_current_game],
                    data_stockfish_estimation_state['Result'][number_of_current_game])
                )
                plt.savefig(path_to_folder_to_save_graphics + str(number_of_current_game) + '.png')

        stats_df: pd.DataFrame = pd.DataFrame(data_stockfish_estimation_state)
        return stats_df

    def diff_value_input_state_vs_subgolas_one_game(
            self,
            game: OneGameData,
            number_of_subgoals: int,
            path_to_folder_to_save_graphics: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Function takes chess game, then generates subgoals form each state (from each board during play) and evaluates
        by stockfish.

        :param: game: OneGameData data type (consists of game information and trajectory).
        :param: number_of_subgoals: Number of subgoals we want to generate for current state.
        :param: path_to_folder_to_save_graphics:
        :return: Dataframe which consits of evaluations (by stockfish) entire game with subgoals. Each row represents
        current evaluated state wwith subgoals form it.

        For example:
                                    Evaluation of current state  Subgoal_1 evaluation  Subgoal_2 evaluation
                        0                           105                    75                    49
                        1                             2                     4                    35
                        2                            73                    75                    61
                        3                           -16                    59                    20
                        4                            66                    76                    60
                        5                            34                    33                   -31
                        6                            47                    61                   115
                        7                            73                    49                   115
                        8                            15                    15                    79
                        9                            90                   504                    90
                        10                           17                    22                    17
                        11                          106                   106                    58
                        12                           12                     7                    12
                        13                           92                    92                    78
                        14                           10                    10                     8
                        15                          138                   138                    89
        """

        game_info: Dict[str, str] = game[0].__dict__  # some information that will be use in the future
        game_trajectory: List[Transition] = game[1]
        data_stockfish_estimation_state: Dict[str, List] = {'Evaluation of current state': []}
        collect_data_to_image: Dict[int, List[Tuple[ImmutableBoard, float]]] = {}
        
        for i in range(1, number_of_subgoals + 1):
            data_stockfish_estimation_state['Subgoal_' + str(i) + ' evaluation'] = []

        for row, transition in tqdm(enumerate(game_trajectory)):
            collect_data_to_image[row] = []
            current_game_state: ImmutableBoard = transition[0]
            subgoals_from_current_game_state: List[ImmutableBoard] = \
                self.subgoal_model.generate_subgoals(current_game_state, number_of_subgoals)

            for p, eval_name in enumerate(data_stockfish_estimation_state):
                if eval_name == 'Evaluation of current state':
                    board_after_n_moves_eval: float = \
                        evaluate_immutable_board_by_stockfish_with_resret_machine(current_game_state)
                    data_stockfish_estimation_state[eval_name].append(board_after_n_moves_eval)
                    collect_data_to_image[row].append((current_game_state, board_after_n_moves_eval))
                else:
                    subgoals_from_board_after_n_moves_eval: float = \
                        evaluate_immutable_board_by_stockfish_with_resret_machine(
                            subgoals_from_current_game_state[p - 1]
                        )
                    data_stockfish_estimation_state[eval_name].append(subgoals_from_board_after_n_moves_eval)
                    collect_data_to_image[row].append(
                        (subgoals_from_current_game_state[p - 1], subgoals_from_board_after_n_moves_eval)
                    )

        if isinstance(path_to_folder_to_save_graphics, str):
            pass

        stats_df: pd.DataFrame = pd.DataFrame(data_stockfish_estimation_state)
        return stats_df
