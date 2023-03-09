from copy import deepcopy
from typing import List

from tqdm import tqdm

from utils.data_utils import immutable_boards_to_img
from data_processing.archive.pgn.mcts_data_generator import SubgoalMCGamesDataGenerator
from jobs.core import Job

from metric_logging import log_object, log_value
from policy.cllp import CLLP


class EvaluateCLLP(Job):
    def __init__(
        self,
        k: List[int],
        n_subgoals: int,
        cllp_checkpoint: str,
        cllp_model: str,
        trees_file_path: str = None,
        n_eval_datapoints: int = 1000,
        n_log_samples: int = 10,
        eval_batch_size: int = 1,
        num_beams: int = 16,
        num_return_sequences: int = 2
    ):

        self.k = k
        self.num_beams = num_beams
        self.num_return_sequences = num_return_sequences

        if cllp_model == "all_moves":
            self.cllp = CLLP(cllp_checkpoint, self.num_beams, self.num_return_sequences)
        if cllp_model == "one_move":
            self.cllp = CLLPOneMove(cllp_checkpoint)

        self.eval_data = {}

        self.eval_batch_size = eval_batch_size

        for k in self.k:
            data_generator = SubgoalMCGamesDataGenerator(
                k, n_subgoals, n_eval_datapoints, input_data_dir=trees_file_path, log_samples_limit=0
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
        self.stats["global_reached"] = 0
        self.stats["global_exact_match"] = 0
        self.stats["global_samples"] = 0

    def execute_k(self, k):
        batches_to_predict = []
        current_batch = []
        for sample in tqdm(self.eval_data[k]):
            current_batch.append(deepcopy(sample))
            if len(current_batch) == self.eval_batch_size:
                batches_to_predict.append(current_batch)
                current_batch = []

        for batch in tqdm(batches_to_predict):
            list_of_queries = []
            for sample in batch:
                list_of_queries.append((sample["input_immutable_board"], sample["target_immutable_board"]))

            cllp_moves_batch = self.cllp.get_paths_batch(list_of_queries)

            for num, sample in enumerate(batch):
                cllp_moves = cllp_moves_batch[num]
                input_board = sample["input_immutable_board"]
                target_board = sample["target_immutable_board"]
                try:
                    result = self.push_moves(cllp_moves[0], sample["leela_moves"], input_board, target_board)
                    if not result["reached"]:
                        result = self.push_moves(cllp_moves[1], sample["leela_moves"], input_board, target_board)
                    self.stats["global_samples"] += 1
                    self.stats[k]["n_samples"] += 1
                    self.stats[k]["reached"] += result["reached"]
                    self.stats[k]["exact_match"] += result["exact_match"]
                    self.stats[k]["errors"] += result["error"]
                    self.stats["global_reached"] += result["reached"]
                    self.stats["global_exact_match"] += result["exact_match"]
                except:
                    self.stats[k]["errors"] += 1
                    self.stats["global_samples"] += 1
                    self.stats[k]["n_samples"] += 1

    def execute(self):
        for k in self.k:
            self.execute_k(k)

        self.get_stats()
        for key, val in self.stats.items():
            if isinstance(val, dict):
                for k, v in val.items():
                    log_value(f"CLLP_{key}_{k}", 0, v)
            else:
                log_value(key, 0, val)
            print(f"{key} = {val}")

        print(self.stats)

    def get_stats(self):
        self.stats["global_reached"] /= self.stats["global_samples"]
        self.stats["global_exact_match"] /= self.stats["global_samples"]
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

        return {
            "reached": board.fen() == target_immutable_board.fen(),
            "exact_match": [str(x) for x in cllp_moves] == true_moves,
            "error": error,
        }
