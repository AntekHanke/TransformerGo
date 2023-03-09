import time

import numpy as np

from data_structures.data_structures import ImmutableBoard
from policy.chess_policy import ChessPolicy
from policy.cllp import CLLP
from subgoal_generator.subgoal_generator import BasicChessSubgoalGenerator
from value.chess_value import ChessValue


def verify_path(input_immutable_board, subgoal, path):
    board = input_immutable_board.to_board()
    for move in path:
        if not board.is_legal(move):
            return False
        board.push(move)
        if ImmutableBoard.from_board(board) == subgoal:
            return True
    return False


class ChessStateExpander:
    def __init__(
        self,
        chess_policy: ChessPolicy,
        chess_value: ChessValue,
        subgoal_generator: BasicChessSubgoalGenerator,
        cllp: CLLP,
    ):
        self.policy = chess_policy
        self.value = chess_value
        self.subgoal_generator = subgoal_generator
        self.cllp = cllp

    def expand_state(
        self,
        input_immutable_board: ImmutableBoard = None,
        cllp_num_beams: int = None,
        cllp_num_return_sequences: int = None,
        generator_num_beams: int = None,
        generator_num_subgoals: int = None,
        sort_subgoals_by: str = None,
        **subgoal_generation_kwargs,
    ):

        subgoals, generation_stats = self.subgoal_generator.generate_subgoals(
            input_immutable_board, generator_num_beams, generator_num_subgoals, **subgoal_generation_kwargs
        )

        subgoals = [subgoal for subgoal in subgoals if subgoal != input_immutable_board]

        paths, cllp_stats = self.cllp.get_paths_batch(
            [(input_immutable_board, subgoal) for subgoal in subgoals], cllp_num_beams, cllp_num_return_sequences
        )

        analysis_time_start = time.time()
        subgoals_info = {}
        for subgoal, paths_to_subgoal in zip(subgoals, paths):
            subgoal_info = self.analyze_subgoal(input_immutable_board, subgoal, paths_to_subgoal)
            if subgoal_info is not None:
                subgoals_info[subgoal] = subgoal_info

        sorted_subgoals = self.sort_subgoals(subgoals_info, sort_subgoals_by)
        stats = dict(**generation_stats, **cllp_stats)
        stats["analysis_time"] = time.time() - analysis_time_start
        return sorted_subgoals, stats

    @staticmethod
    def sort_subgoals(subgoals_info, sort_subgoals_by: str = None):
        if sort_subgoals_by in ["highest_min_probability", "highest_max_probability", "highest_total_probability"]:
            return sorted(
                subgoals_info.keys(),
                key=lambda subgoal: subgoals_info[subgoal][sort_subgoals_by],
                reverse=True,
            )
        else:
            raise ValueError(f"Unknown sort_subgoals_by: {sort_subgoals_by}")

    def analyze_subgoal(self, input_immutable_board, subgoal, paths_to_subgoal):
        correct_paths = [path for path in paths_to_subgoal if verify_path(input_immutable_board, subgoal, path)]
        if len(correct_paths) == 0:
            return None

        paths_raw_probabilities = [
            self.policy.get_path_probabilities(input_immutable_board, path) for path in correct_paths
        ]

        paths_stats = [
            self.analyze_path_probabilities(path_raw_probabilities)
            for path_raw_probabilities in paths_raw_probabilities
        ]

        return {
            "value": self.value.evaluate_immutable_board(subgoal),
            "num_paths": len(paths_to_subgoal),
            "paths": correct_paths,
            "path_raw_probabilities": paths_raw_probabilities,
            "path_probabilities": paths_stats,
            "average_path_probability": np.mean([path_stats["total_path_probability"] for path_stats in paths_stats]),
            "path_with_highest_min_probability": paths_to_subgoal[
                np.argmax([path_stats["min_path_probability"] for path_stats in paths_stats])
            ],
            "path_with_highest_total_probability": paths_to_subgoal[
                np.argmax([path_stats["total_path_probability"] for path_stats in paths_stats])
            ],
            "highest_min_probability": max([path_stats["min_path_probability"] for path_stats in paths_stats]),
            "highest_max_probability": max([path_stats["max_path_probability"] for path_stats in paths_stats]),
            "highest_total_probability": max([path_stats["total_path_probability"] for path_stats in paths_stats]),
        }

    def analyze_path_probabilities(self, paths_raw_probabilities):
        total_path_probability = np.prod(paths_raw_probabilities)
        min_path_probability = min(paths_raw_probabilities)
        max_path_probability = max(paths_raw_probabilities)
        return {
            "total_path_probability": total_path_probability,
            "min_path_probability": min_path_probability,
            "max_path_probability": max_path_probability,
        }
