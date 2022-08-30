import chess
import chess.pgn
import chess.engine

import pandas as pd
import matplotlib.pyplot as plt

from typing import Optional, List, Dict, Tuple
from os import makedirs
import random

from data_processing.data_utils import immutable_boards_to_img, get_split
from data_structures.data_structures import ImmutableBoard, OneGameData, ChessMetadata, Transition
from subgoal_generator.subgoal_generator import BasicChessSubgoalGenerator
from chess_engines.stockfish import evaluate_immutable_board_by_stockfish_with_resret_machine
from data_processing.chess_data_generator import NoFilter, ChessFilter


def n_forwad_moves_chess_board(moves_during_play: List[Tuple[str, chess.Move]],
                               number_of_forward_moves: int) -> Tuple[chess.Board, str]:
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


class ChessStatsEvalDatasetCreator:

    def __init__(self,
                 pgn_file: str,
                 n_games: int,
                 chess_filter: ChessFilter = NoFilter,
                 train_eval_split: float = 0.95,
                 ) -> None:

        self.pgn_database = open(pgn_file, errors="ignore")
        self.n_games = n_games
        self.chess_filter = chess_filter
        self.train_eval_split = train_eval_split

        # list contains chess games that goes (boards from move's trajectory)
        # into the evaluation dataset
        self.dataset_containig_games_to_eval_statistics: Dict[int, Transition] = {}
        self.current_game: Optional[chess.pgn.Game] = None

    def next_game_to_raw_data(self) -> Optional[OneGameData]:

        self.current_game: chess.pgn.Game = chess.pgn.read_game(self.pgn_database)

        if self.current_game is None:  # condition is met if there are no more games in the dataset
            return None
        else:
            chess_metadata: ChessMetadata = ChessMetadata(**self.current_game.headers)
            transitions: List[Transition] = []

            if self.chess_filter.use_game(chess_metadata):
                board = chess.Board()
                for move in enumerate(self.current_game.mainline_moves()):
                    _, chess_move = move
                    try:
                        transitions.append(Transition(ImmutableBoard.from_board(board), chess_move))
                        board.push(chess_move)
                    except:
                        break

            return OneGameData(chess_metadata, transitions)

    def create_data(self) -> None:

        n_iterations: int = 0
        dict_position: int = 0

        while self.n_games > 0:
            n_iterations += 1
            train_eval: str = get_split(n_iterations, self.train_eval_split)

            if train_eval == "eval":
                game = self.next_game_to_raw_data()
                if len(game.transitions) == 0:  # empty games created after applying the filter
                    continue
                else:
                    self.dataset_containig_games_to_eval_statistics[dict_position] = game
                    dict_position += 1
                    self.n_games -= 1
            else:
                game = self.next_game_to_raw_data()

            if game is None:  # the entire dataset was used
                break


class StatisticOfSubgoals:
    def __init__(self,
                 chess_stats_eval_dataset_creator: ChessStatsEvalDatasetCreator,
                 chess_subgaols_generator_checkpoint_path: str) -> None:
        """
        Atribut self.chess_dataset_statistics is a pandas frame which contains information about chess dataset.

        :param: database_of_chess_games_path: This is path to dataset which contain chess gameplays.
        Dataset must have .pgn format.
        :param: chess_subgaols_generator_checkpoint_path: This is path to transformer's model.
        """

        self.chess_stats_eval_dataset_creator = chess_stats_eval_dataset_creator
        self.chess_subgaols_generator_checkpoint_path = chess_subgaols_generator_checkpoint_path
        self.subgoal_model: BasicChessSubgoalGenerator = \
            BasicChessSubgoalGenerator(self.chess_subgaols_generator_checkpoint_path)

       # self.chess_dataset_statistics: pd.DataFrame = self.chess_dataset_stats()  # TODO: consider filters !

    def chess_dataset_stats(self, path_to_save_statistics: Optional[str] = None) -> pd.DataFrame:
        """
        A functions that counts the number of games, number of games, the number of games the white opponent has won,
        the number of games the black opponent has won and the number of games and the number of games ended in a draw.

        For exaple:

          Number of games:  Number of games won by white player:  Number of games won by black player:  Nuber of draws:
        0               33                                   15                                      6                12

        :param: path_to_save_statistics: Path to save chess dataset (.csv format). If path_to_save_statistics is None,
        then the data is not saved (by defult is None), e.g. 'path/to/save/stats/name_of_document.csv.
        :return: Pandas frame which contains dataset statistic information
        """

        number_of_games: int = 0
        white_won: int = 0
        black_won: int = 0
        drawns: int = 0
        database_statistic: dict = {}
        database_of_chess_games_file = open(self.database_of_chess_games_path)  # TODO: missing typing

        while True:
            game = chess.pgn.read_game(database_of_chess_games_file)
            if game is None:
                break
            else:
                number_of_games += 1
                game_result = game.headers["Result"]
                if game_result == '1-0':
                    white_won += 1
                elif game_result == '0-1':
                    black_won += 1
                else:
                    drawns += 1
        assert number_of_games == white_won + black_won + drawns, 'number_of_games = white_won + black_won + draws'

        database_statistic['Number of games: '] = [number_of_games]
        database_statistic['Number of games won by white player: '] = [white_won]
        database_statistic['Number of games won by black player: '] = [black_won]
        database_statistic['Nuber of draws: '] = [drawns]

        df = pd.DataFrame(database_statistic)
        database_of_chess_games_file.close()

        if path_to_save_statistics is not None:
            makedirs(path_to_save_statistics, exist_ok=True)
            df.to_csv(path_to_save_statistics)

        return df

    def diff_value_input_state_vs_subgolas_multi_games(self,
                                                       number_of_subgoals: int,
                                                       path_to_folder_to_save_graphics: Optional[str] = None
                                                       ) -> pd.DataFrame:

        database_of_chess_games: Dict[int, Transition] = \
            self.chess_stats_eval_dataset_creator.dataset_containig_games_to_eval_statistics

        data_stockfish_estimation_state: dict = {'Input board evaluation': []}
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

        eval_boards_key_names: List[str] = \
            ['Input board evaluation'] + ['Subgoal_' + str(i) + ' evaluation' for i in range(1, number_of_subgoals + 1)]

        for number_of_current_game in range(len(database_of_chess_games)):
            current_game: OneGameData = database_of_chess_games[number_of_current_game]
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

            if isinstance(path_to_folder_to_save_graphics, str):
                immutable_boards_to_img([immutable_board_after_n_moves]
                                        + subgoals_from_board_after_n_moves,
                                        [name + ': ' + str(data_stockfish_estimation_state[name][number_of_current_game])
                                         for name in eval_boards_key_names]
                                        )
                plt.savefig(path_to_folder_to_save_graphics + str(number_of_current_game) + '.png')

        stats_df: pd.DataFrame = pd.DataFrame(data_stockfish_estimation_state)
        return stats_df

    def diff_value_input_state_vs_subgolas_one_game(self) -> pd.DataFrame:
        pass


if __name__ == '__main__':
    path_to_chess_dataset = '/home/gracjan/subgoal/subgoal_search_chess/assets/cas_small.pgn'
    path_to_subgoal_generator = '/home/gracjan/subgoal/subgoal_search_chess/generator_checkpoints/k_2/checkpoint-2000'
    path_to_save_stockfish_image_evaluations = '/home/gracjan/subgoal/subgoal_search_chess/example_of_stats/'

    dataset = ChessStatsEvalDatasetCreator(pgn_file=path_to_chess_dataset, n_games=3, train_eval_split=0.5)
    dataset.create_data()
    stats = StatisticOfSubgoals(dataset, path_to_subgoal_generator)
    print(stats.diff_value_input_state_vs_subgolas_multi_games(3, path_to_save_stockfish_image_evaluations).to_string())

