import random
from typing import List

import pandas as pd
from stockfish import Stockfish

from chess_engines.stockfish import StockfishEngine
from data_processing.chess_data_generator import ChessDataGenerator, NoFilter, ChessSubgoalDataGenerator
from data_processing.chess_tokenizer import ChessTokenizer
from data_processing.data_utils import immutable_boards_to_img, RESULT_TO_WINNER
from data_processing.exhaustive_search import ExhaustiveSearch
from jobs.core import Job
from transformers import (
    Trainer,
    BartForConditionalGeneration,
    BartConfig,
    TrainingArguments,
)

from metric_logging import log_param, source_files_register, log_object, log_value
from statistics.statistics_dataset_generator import StatisticsDatasetCreator
from subgoal_generator.subgoal_generator import ChessSubgoalGenerator, BasicChessSubgoalGenerator

source_files_register.register(__file__)


class EvaluateModule:
    def evaluate_list_of_examples(self, list_of_examples):
        pass

    def evaluate_example(self, input_board, list_of_subgoals):
        pass


class SubgoalQualityDatabaseGenerator(Job):
    def __init__(
        self,
        k: int,
        n_subgoals: int,
        n_games: int,
        pgn_file: str,
        subgoal_generator: ChessSubgoalGenerator,
        take_transition_p: float = 0.05,
        n_eval_datapoints: int = 1000,
        check_exhaustive_search: bool = True,
        top_n_actions_range: range = range(1, 4),
    ):

        self.k = k
        self.n_subgoals = n_subgoals
        self.n_games = n_games

        self.subgoal_generator = subgoal_generator
        self.take_transition_p = take_transition_p
        self.chess_database = StatisticsDatasetCreator(pgn_file, self.n_games)
        self.n_eval_datapoints = n_eval_datapoints
        self.check_exhaustive_search = check_exhaustive_search
        self.top_n_actions_range = top_n_actions_range

        self.evaluated_transitions = 0

    def execute(self):
        self.chess_database.create_data()
        random_games_order = list(range(len(self.chess_database.games_to_eval)))
        data_rows = []
        for game_idx in random_games_order:
            if self.evaluated_transitions >= self.n_eval_datapoints:
                break
            data_rows.extend(self.eval_one_game(self.chess_database.games_to_eval[game_idx]))


            log_value("evaluated_transitions", game_idx, self.evaluated_transitions)
            log_value("evaluated_games", game_idx, game_idx + 1)
        return pd.DataFrame(data_rows)

    def eval_one_game(self, one_game_data):
        data_rows = []
        for transition_num in range(len(one_game_data.transitions)):
            if random.random() < self.take_transition_p:
                data_rows.append(self.eval_transition(transition_num, one_game_data))
        return data_rows

    def eval_transition(self, transition_num, one_game_data):
        transition = one_game_data.transitions[transition_num]
        subgoals = self.generate_subgoals(transition.immutable_board)
        target_idx = min(transition_num + self.k, len(one_game_data.transitions) - 1)
        target_board = one_game_data.transitions[target_idx].immutable_board

        row_data = {
            "n_subgoals": len(subgoals),
            "input_board_fen": transition.immutable_board.fen(),
            "target_board_fen": target_board.fen(),
            "result": one_game_data.metadata.Result,
            "winner": RESULT_TO_WINNER[one_game_data.metadata.Result],
            "move_num": transition_num,
            "game_len": len(one_game_data.transitions),
            "input_board_value": StockfishEngine.evaluate_immutable_board(transition.immutable_board),
            "target_board_value": StockfishEngine.evaluate_immutable_board(target_board),
        }

        for id, subgoal in enumerate(subgoals):
            row_data[f"subgoal_{id}_fen"] = subgoal.fen()
            row_data[f"subgoal_{id}_value"] = StockfishEngine.evaluate_immutable_board(subgoal)

        if self.check_exhaustive_search:
            search_wit_all_actions = ExhaustiveSearch(transition.immutable_board, self.k, None)
            search_wit_top_n_actions = [
                ExhaustiveSearch(transition.immutable_board, self.k, n) for n in self.top_n_actions_range
            ]

            for num, accessible in enumerate(search_wit_all_actions.check_subgoals(subgoals)["accessible"]):
                row_data[f"subgoal_{num}_accessible"] = accessible

            for num, distance in enumerate(search_wit_all_actions.check_subgoals(subgoals)["distance"]):
                row_data[f"subgoal_{num}_distance"] = distance


            for top_n_actions in self.top_n_actions_range:
                subgoals_accessible = search_wit_top_n_actions[
                    top_n_actions - 1
                ].check_subgoals(subgoals)["accessible"]

                for id, subgoal_accessible in enumerate(subgoals_accessible):
                    row_data[f"subgoal_{id}_accessible_top_{top_n_actions}"] = subgoal_accessible

        self.evaluated_transitions += 1
        return row_data

    def generate_subgoals(self, input_board):
        subgoals = self.subgoal_generator.generate_subgoals(input_board, self.n_subgoals)
        for subgoal in subgoals:
            assert subgoal.board != input_board.board, "Subgoal is the same as input board"
        return subgoals


