import chess
import chess.pgn
import chess.engine

import pandas as pd
import matplotlib.pyplot as plt

from typing import Optional, List
from os import makedirs
import random

from data_processing.data_utils import immutable_boards_to_img
from data_structures.data_structures import ImmutableBoard
from subgoal_generator.subgoal_generator import BasicChessSubgoalGenerator
from chess_engines.stockfish import evaluate_immutable_board_by_stockfish_with_resret_machine


def odd_even_random_move(moves_during_play: chess.pgn.Mainline, white_or_black: str) -> int:
    """
    For example: if we choose '1-0' and the current moves are =
    1. e4 c5 2. f4 d6 3. Nf3 Nc6 4. Bb5 Bd7 5. c3 a6 6. Ba4 b5 7. Bc2 c4 8. d4 cxd3 9. Qxd3 g6 10. Be3 Bg7 11. O-O Nf6,
    then we randomly choose odd number from 1 to 11 (all moves), e.g. 5.

    :param: white_or_black: '1-0' if white player won, '0-1' if black player won.
    :param: moves_during_play: These are all moves during current playgame.
    :return: Number of moves ahed from starting board.
    """
    number_of_moves: int = 0
    moves: List[chess.Move] = [move for move in moves_during_play]

    if white_or_black == '1-0':
        number_of_moves = random.randrange(1, len(moves), 2)
    elif white_or_black == '0-1':
        number_of_moves = random.randrange(2, len(moves), 2)
    return number_of_moves


def n_forwad_moves_chess_board(moves_during_play: chess.pgn.Mainline, number_of_forward_moves: int) -> chess.Board:
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
    beggining_board: chess.Board = chess.Board()
    moves: List[chess.Move] = [move for move in moves_during_play]
    number_of_forward_moves: int = min(number_of_forward_moves, len(moves))

    for i in range(number_of_forward_moves):
        beggining_board.push(moves[i])

    return beggining_board


class QualityOfSubgoals:
    def __init__(self,
                 database_of_chess_games_path: str,
                 chess_subgaols_generator_checkpoint_path: str) -> None:
        """
        Atribut self.chess_dataset_statistics is a pandas frame which contains information about chess dataset.

        :param: database_of_chess_games_path: This is path to dataset which contain chess gameplays.
        Dataset must have .pgn format.
        :param: chess_subgaols_generator_checkpoint_path: This is path to transformer's model.
        """

        self.database_of_chess_games_path = database_of_chess_games_path
        self.chess_subgaols_generator_checkpoint_path = chess_subgaols_generator_checkpoint_path
        self.subgoal_model: BasicChessSubgoalGenerator = \
            BasicChessSubgoalGenerator(self.chess_subgaols_generator_checkpoint_path)
        self.chess_dataset_statistics: pd.DataFrame = self.chess_dataset_stats()

    def chess_dataset_stats(self, path_to_save_statistics: Optional[str] = None) -> pd.DataFrame:
        """
        A functions that counts the number of games, number of games, the number of games the white opponent has won,
        the number of games the black opponent has won and the number of games and the number of games ended in a draw.

        For exaple:

          Number of games:  Number of games won by white player:  Number of games won by black player:  Nuber of draws:
        0               33                                   15                                      6                12

        :param: path_to_save_statistics: Path to save chess dataset (.csv format). If path_to_save_statistics is None,
        then the data is not saved (by default is None), e.g. 'path/to/save/stats/name_of_document.csv.
        :return: Pandas frame which contains dataset statistic information.
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

        df: pd.DataFrame = pd.DataFrame(database_statistic)
        database_of_chess_games_file.close()

        if path_to_save_statistics is not None:
            makedirs(path_to_save_statistics, exist_ok=True)
            df.to_csv(path_to_save_statistics)

        return df

    def diff_value_input_state_vs_subgolas_multi_games(self,
                                                       number_of_games: int,
                                                       white_or_black: str,
                                                       number_of_subgoals: int,
                                                       path_to_folder_to_save_graphics: Optional[str] = None
                                                       ) -> pd.DataFrame:
        """
        This function returns Pandas dataframe which contains information about stockfish value estiamtion
        on the input board and n subgoals (form imput board) in n games.

        For example_1: For the 3 games (number_of_games = 3) that were won by white oponent (white_or_black = '1-0')
        and with 2 subgols form each input board, we get the following table:

                Input board evaluation  Subgoal_1 evaluation  Subgoal_2 evaluation
            0                     -31                    22                  -114
            1                    -182                  -179                   -58
            2                    -105                    21                  -111

        Important note: If subgoal generator was trained on the boards that were won by white oponent
        (white_or_black = '1-0'), then we give to the generator the boards (states) in which the black
        player is to make a move.

        For example_2:
        We have a generator trained on the boards that the white opponents (1-0) won.Then we take the gameplay,
        draw the number of moves and from the resulting state we generate the subgoals (the resulting state is the state
        in which the black player has move).

        :param: number_of_games: The number of games we want to evaluate (have in the table in the table).
        :param: white_or_black: Type of games on which the generator was trained (which we select from a chees dataset).
        :param: number_of_subgoals: Number of subgols from giveb state.
        :param: path_to_folder_to_save_graphics: If path_to_folder_to_save_graphics is not None (default is None), then each row of
         dataframe is saved with images of the boards.
        :return: Dataframe with stats.
        """

        assert number_of_games > 0 and isinstance(number_of_games, int), "Number of games must be positive integer."

        data_stockfish_estimation_state: dict = {'Input board evaluation': []}
        game: int = 0  # number of current game

        if white_or_black == '1-0':
            number_of_games = min(number_of_games,
                                  self.chess_dataset_statistics['Number of games won by white player: '][0])
        elif white_or_black == '0-1':
            number_of_games = min(number_of_games,
                                  self.chess_dataset_statistics['Number of games won by black player: '][0])

        database_of_chess_games = open(self.database_of_chess_games_path)

        for i in range(1, number_of_subgoals + 1):
            data_stockfish_estimation_state['Subgoal_' + str(i) + ' evaluation'] = []

        for _ in range(self.chess_dataset_statistics['Number of games: '][0]):
            game_from_dataset = chess.pgn.read_game(database_of_chess_games)
            if game_from_dataset.headers['Result'] == white_or_black:
                moves_in_current_game: chess.pgn.Mainline = game_from_dataset.mainline_moves()
                number_of_forward_moves: int = odd_even_random_move(moves_in_current_game, white_or_black)
                board_after_n_moves: chess.Board = n_forwad_moves_chess_board(moves_in_current_game,
                                                                              number_of_forward_moves
                                                                              )
                immutable_board_after_n_moves: ImmutableBoard = ImmutableBoard.from_board(board_after_n_moves)
                subgoals_from_board_after_n_moves: List[ImmutableBoard] = \
                    self.subgoal_model.generate_subgoals(immutable_board_after_n_moves, number_of_subgoals)

                for p, key in enumerate(data_stockfish_estimation_state.keys()):
                    if key == 'Input board evaluation':
                        board_after_n_moves_eval: float = \
                            evaluate_immutable_board_by_stockfish_with_resret_machine(immutable_board_after_n_moves)
                        data_stockfish_estimation_state[key].append(board_after_n_moves_eval)
                    else:
                        subgoals_from_board_after_n_moves_eval: float = \
                            evaluate_immutable_board_by_stockfish_with_resret_machine(
                                subgoals_from_board_after_n_moves[p - 1])
                        data_stockfish_estimation_state[key].append(subgoals_from_board_after_n_moves_eval)

                if isinstance(path_to_folder_to_save_graphics, str):
                    immutable_boards_to_img([ImmutableBoard.from_board(board_after_n_moves)]
                                            + subgoals_from_board_after_n_moves,
                                            [name + ': ' + str(value[game]) for name, value in data_stockfish_estimation_state.items()]
                                            )
                    plt.savefig(path_to_folder_to_save_graphics + str(game) + '.png')
                    game = game + 1

                number_of_games -= 1
                if number_of_games == 0:
                    break

        database_of_chess_games.close()
        stats_df: pd.DataFrame = pd.DataFrame(data_stockfish_estimation_state)
        return stats_df

    def diff_value_input_state_vs_subgolas_one_game(self) -> pd.DataFrame:
        pass


if __name__ == '__main__':
    path_to_chess_dataset = '/home/gracjan/subgoal/subgoal_search_chess/assets/cas_small.pgn'
    path_to_subgoal_generator = '/home/gracjan/subgoal/subgoal_search_chess/generator_checkpoints/k_2/checkpoint-2000'
    path_to_save_stockfish_image_evaluations = '/home/gracjan/subgoal/subgoal_search_chess/example_of_stats/'

    stats = QualityOfSubgoals(path_to_chess_dataset, path_to_subgoal_generator)
    print(stats.chess_dataset_statistics.to_string())

    stockfish_stats = stats.diff_value_input_state_vs_subgolas_multi_games(number_of_games=3,
                                                                           white_or_black='1-0',
                                                                           number_of_subgoals=2,
                                                                           path_to_folder_to_save_graphics=path_to_save_stockfish_image_evaluations
                                                                           )
    print(stockfish_stats.to_string())
