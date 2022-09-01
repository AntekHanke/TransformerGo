import random
from typing import List

import pandas as pd
from stockfish import Stockfish

from chess_engines.stockfish import StockfishEngine
from data_processing.chess_data_generator import ChessDataGenerator, NoFilter, ChessSubgoalDataGenerator
from data_processing.chess_tokenizer import ChessTokenizer
from data_processing.data_utils import immutable_boards_to_img, RESULT_TO_WINNER
from jobs.core import Job
from transformers import (
    Trainer,
    BartForConditionalGeneration,
    BartConfig,
    TrainingArguments,
)

from metric_logging import log_param, source_files_register, log_object
from mrunner_utils.neptune_logger import NeptuneLogger
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
        n_log_samples: int = 100,
    ):

        self.k = k
        self.n_subgoals = n_subgoals
        self.n_games = n_games

        self.subgoal_generator = subgoal_generator
        self.take_transition_p = take_transition_p
        self.chess_database = StatisticsDatasetCreator(pgn_file, self.n_games)
        self.n_eval_datapoints = n_eval_datapoints
        self.n_log_samples = n_log_samples

        self.evaluated_transitions = 0

    def execute(self):
        self.chess_database.create_data()
        random_games_order = list(range(len(self.chess_database.games_to_eval)))
        data_rows = []
        for game_idx in random_games_order:
            if self.evaluated_transitions >= self.n_eval_datapoints:
                break
            data_rows.extend(self.eval_one_game(self.chess_database.games_to_eval[game_idx]))
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
        target_idx = min(transition_num + self.k, len(self.chess_database.games_to_eval[0].transitions) - 1)
        target_board = one_game_data.transitions[target_idx].immutable_board

        row_data = {
            "n_subgoals": len(subgoals),
            "input_board_fen": transition.immutable_board.fen(),
            "target_board_fen": target_board.fen(),
            "result": one_game_data.metadata.Result,
            "winner": RESULT_TO_WINNER[one_game_data.metadata.Result],
            "input_board_value": StockfishEngine.evaluate_immutable_board(transition.immutable_board),
            "target_board_value": StockfishEngine.evaluate_immutable_board(target_board),
            "subgoals_values": [StockfishEngine.evaluate_immutable_board(subgoal) for subgoal in subgoals],
        }

        self.evaluated_transitions += 1
        return row_data


    def generate_subgoals(self, input_board):
        subgoals = self.subgoal_generator.generate_subgoals(input_board, self.n_subgoals)
        for subgoal in subgoals:
            assert subgoal.board != input_board.board, "Subgoal is the same as input board"
        return subgoals

    def evaluate_with_stockfish(
        self,
        input_board,
        subgoals,
    ):
        pass


duper = SubgoalQualityDatabaseGenerator(
    2,
    3,
    500,
    "/home/tomek/Research/subgoal_chess_data/chess_micro_aa",
    BasicChessSubgoalGenerator("/home/tomek/Research/subgoal_chess_data/generator_k_2/out/checkpoint-2000"),
    take_transition_p=0.1,
    n_eval_datapoints=100,
)

df = duper.execute()

print(df)
