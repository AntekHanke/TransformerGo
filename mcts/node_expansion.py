import time
from typing import Type, List

import numpy as np
from chess import Move

from data_structures.data_structures import ImmutableBoard
from metric_logging import log_value_to_average, log_value_to_accumulate
from policy.chess_policy import ChessPolicy
from policy.cllp import CLLP
from subgoal_generator.subgoal_generator import BasicChessSubgoalGenerator
from value.chess_value import ChessValue


def verify_path(input_immutable_board: ImmutableBoard, subgoal: ImmutableBoard, path: List[Move]) -> List[Move]:
    board = input_immutable_board.to_board()
    correct_path = []
    for move in path:
        if not board.is_legal(move):
            return None
        board.push(move)
        correct_path.append(move)
        if ImmutableBoard.from_board(board) == subgoal:
            return correct_path
    return None


class ChessStateExpander:
    def __init__(
        self,
        chess_policy_class: Type[ChessPolicy],
        chess_value_class: Type[ChessValue],
        subgoal_generator_class: Type[BasicChessSubgoalGenerator],
        cllp_class: Type[CLLP],
    ):
        self.policy = chess_policy_class()
        self.value = chess_value_class()
        self.subgoal_generator = subgoal_generator_class()
        self.cllp = cllp_class()

    def expand_state(
        self,
        input_immutable_board: ImmutableBoard = None,
        siblings_states: List[ImmutableBoard] = None,
        cllp_num_beams: int = None,
        cllp_num_return_sequences: int = None,
        generator_num_beams: int = None,
        generator_num_subgoals: int = None,
        sort_subgoals_by: str = None,
        **subgoal_generation_kwargs,
    ):

        if not self.subgoal_generator.is_in_memory(input_immutable_board):
            log_value_to_average("generator_used_%", 1)
            log_value_to_accumulate("generator_used", 1)
            self.subgoal_generator.generate_subgoals(
                siblings_states, generator_num_beams, generator_num_subgoals, **subgoal_generation_kwargs
            )
        else:
            log_value_to_average("generator_used_%", 0)
            log_value_to_accumulate("generator_used", 0)

        subgoals = self.subgoal_generator.generate_use_memory(input_immutable_board)

        subgoals = [subgoal for subgoal in subgoals if subgoal != input_immutable_board]

        paths = self.cllp.get_paths_batch(
            [(input_immutable_board, subgoal) for subgoal in subgoals], cllp_num_beams, cllp_num_return_sequences
        )

        analysis_time_start = time.time()
        subgoals_info = {}
        for subgoal, paths_to_subgoal in zip(subgoals, paths):
            subgoal_info = self.analyze_subgoal(input_immutable_board, subgoal, paths_to_subgoal)
            if subgoal_info is not None:
                subgoals_info[subgoal] = subgoal_info

        sorted_subgoals = self.sort_subgoals(subgoals_info, sort_subgoals_by)
        log_value_to_accumulate("analysis_time", time.time() - analysis_time_start)
        log_value_to_average("analysis_time_avg", time.time() - analysis_time_start)

        return sorted_subgoals, subgoals_info

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
        correct_paths = [
            correct_path
            for path in paths_to_subgoal
            if (correct_path := verify_path(input_immutable_board, subgoal, path)) is not None
        ]
        if len(correct_paths) == 0:
            return None

        time_start = time.time()
        paths_raw_probabilities = [
            self.policy.get_path_probabilities(input_immutable_board, path) for path in correct_paths
        ]
        log_value_to_accumulate("paths_raw_probabilities_total", time.time() - time_start)
        log_value_to_average("paths_raw_probabilities_avg", time.time() - time_start)

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
