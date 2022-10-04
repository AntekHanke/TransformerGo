from typing import Dict, Any

import chess
import pandas as pd

from data_processing.chess_data_generator import ChessGamesDataGenerator, ChessFilter, NoFilter
from data_processing.data_utils import get_split
from data_structures.data_structures import Transition, OneGameData, ChessMetadata
from metric_logging import log_value


class StatisticsDatasetCreator(ChessGamesDataGenerator):
    def __init__(
        self, pgn_file: str, n_games: int, train_eval_split: float = 0.95, chess_filter: ChessFilter = NoFilter()
    ):
        super().__init__(pgn_file, chess_filter, train_eval_split)

        self.pgn_file = pgn_file
        self.pgn_database = open(pgn_file, errors="ignore")
        self.n_games = n_games  # number of games that we want to evaluate (we want statistics from them)
        self.chess_filter = chess_filter
        self.train_eval_split = train_eval_split

        # list contains chess games that goes (boards from move's trajectory)
        # into the evaluation dataset
        self.games_to_eval = {}

    def create_data(self) -> None:
        """
        Function splits chess dataset on training and evaluation set (using filter - by default there is no filter).
        Spliting is the same as in ChessDataGenerator (by get_split function). Only evaluation set is saved.
        :return: None
        """
        n_iterations: int = 0
        dict_position: int = 0

        while len(self.games_to_eval) < self.n_games:
            n_iterations += 1
            train_eval: str = get_split(n_iterations, self.train_eval_split)
            game = self.next_game_to_raw_data()

            if game is None:  # The entire dataset was used.
                break

            if train_eval == "eval":
                if len(game.transitions) == 0:  # Empty games created after applying the filter.
                    continue
                else:
                    self.games_to_eval[dict_position] = game
                    dict_position += 1

            if len(self.games_to_eval) % 100 == 0 and len(self.games_to_eval) > 0:
                log_value("stats_data_generation", n_iterations, len(self.games_to_eval))
                log_value("stats_data_progress", n_iterations, len(self.games_to_eval)/self.n_games)

        print(f'Finished creating {len(self.games_to_eval)} games for evaluation.')
    def game_to_datapoints(self, one_game_data: OneGameData, current_dataset: Dict):
        raise NotImplementedError

    def sample_to_log_object(self, sample: Any, game_metadata: ChessMetadata) -> Any:
        raise NotImplementedError

    def chess_dataset_stats(self) -> pd.DataFrame:
        """
          A functions that counts the number of games, the number of games that white opponent has won,
          the number of games that black opponent has won, the number of games ended in a draw and the and
          the number of games filtered by the filter used.

          For example:

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
                game_result = chess_metadata.__dict__["Result"]
                number_of_games += 1

                if self.chess_filter.use_game(chess_metadata):
                    filtred_games += 1

                if game_result == "1-0":
                    white_won += 1
                elif game_result == "0-1":
                    black_won += 1
                else:
                    draws += 1

        assert number_of_games == white_won + black_won + draws, "number_of_games = white_won + black_won + draws"

        database_statistic["Number of games: "] = [number_of_games]
        database_statistic["Number of games won by white player: "] = [white_won]
        database_statistic["Number of games won by black player: "] = [black_won]
        database_statistic["Number of draws: "] = [draws]
        database_statistic["Number of filtered games: "] = [filtred_games]

        df_chess_games_dataset_stats = pd.DataFrame(database_statistic)
        database_of_chess_games_file.close()

        return df_chess_games_dataset_stats
