import random
from typing import List

import chess
from tqdm import tqdm

from chess_engines.stockfish import StockfishEngine
from data_processing.chess_data_generator import ChessGamesDataGenerator, NoFilter, ChessSubgoalGamesDataGenerator
from data_processing.chess_tokenizer import ChessTokenizer
from data_processing.data_utils import immutable_boards_to_img
from data_processing.mcts_data_generator import SubgoalMCGamesDataGenerator
from jobs.core import Job
from transformers import (
    Trainer,
    BartForConditionalGeneration,
    BartConfig,
    TrainingArguments,
)

from leela.leela_graph_data_loader import data_trees_generator
from metric_logging import log_param, source_files_register, log_object, log_value
from policy.cllp import CLLP
from subgoal_generator.subgoal_generator import ChessSubgoalGenerator

source_files_register.register(__file__)


class EvaluateCLLP(Job):
    def __init__(
        self,
        k: List[int],
        n_subgoals: int,
        cllp_checkpoint: str,
        trees_file_path: str = None,
        n_eval_datapoints: int = 1000,
        n_log_samples: int = 10,
    ):

        self.k = k
        self.cllp = CLLP(cllp_checkpoint)
        self.eval_data = {}

        for k in self.k:
            data_generator = SubgoalMCGamesDataGenerator(
                k, n_subgoals, n_eval_datapoints, input_data_dir=trees_file_path
            )
            data_generator.generate_data()
            self.eval_data[k] = []
            for _, row in data_generator.data.iterrows():
                self.eval_data[k].append(
                    {
                        "input_immutable_board": row["input_immutable_board"],
                        "target_immutable_board": row["target_immutable_board"],
                        "leela_moves": row["moves"],
                    }
                )

        self.n_log_samples = n_log_samples
        self.logged_samples = 0

        self.stats = {k: {"reached": 0, "exact_match": 0, "n_samples": 0, "errors": 0} for k in self.k}
        self.stats['global_reached'] = 0
        self.stats['global_exact_match'] = 0
        self.stats['global_samples'] = 0

    def execute(self):
        for k in self.k:
            for sample in tqdm(self.eval_data[k]):
                input_board = sample["input_immutable_board"]
                target_board = sample["target_immutable_board"]
                cllp_moves = self.cllp.get_path(input_board, target_board)
                result = self.push_moves(cllp_moves, sample["leela_moves"], input_board, target_board)
                self.stats['global_samples'] += 1
                self.stats[k]['n_samples'] += 1
                self.stats[k]["reached"] += result["reached"]
                self.stats[k]["exact_match"] += result["exact_match"]
                self.stats[k]["errors"] += result["error"]
                self.stats['global_reached'] += result["reached"]
                self.stats['global_exact_match'] += result["exact_match"]

                # assert False
        self.get_stats()
        for key, val in self.stats.items():
            if isinstance(val, dict):
                for k, v in val.items():
                    log_value(f"CLLP_{key}_{k}", 0, v)
            else:
                log_value(key, 0, val)
            print(f"{key} = {val}")


    def get_stats(self):
        self.stats['global_reached'] /= self.stats['global_samples']
        self.stats['global_exact_match'] /= self.stats['global_samples']
        for k in self.k:
            self.stats[k]["reached"] /= self.stats[k]["n_samples"]
            self.stats[k]["exact_match"] /= self.stats[k]["n_samples"]


    def push_moves(self, cllp_moves, true_moves, input_immutable_board, target_immutable_board):
        board = input_immutable_board.to_board()
        error = False
        try:
            for move in cllp_moves:
                board.push(move)
        except:
            error = True


        if self.logged_samples < self.n_log_samples:

            fig = immutable_boards_to_img(
                [input_immutable_board, target_immutable_board],
                [f"Predicted moves = {cllp_moves}", f"True moves = {true_moves}"],
            )

            log_object("CLLP", fig)

        return {"reached": board.fen() == target_immutable_board.fen(), "exact_match": [str(x) for x in cllp_moves] == true_moves, "error": error}
