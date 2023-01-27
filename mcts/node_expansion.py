import numpy as np

from data_structures.data_structures import ImmutableBoard


def verify_path(input_immutable_board, subgoal, path):
    # try:
    board = input_immutable_board.to_board()
    for move in path:
        if not board.is_legal(move):
            return False
        board.push(move)
        if ImmutableBoard.from_board(board) == subgoal:
            return True
    return False
    # except:
    #     return False


class ChessStateExpander:
    def __init__(self, chess_policy, chess_value, subgoal_generator, cllp):
        self.policy = chess_policy
        self.value = chess_value
        self.subgoal_generator = subgoal_generator
        self.cllp = cllp

    def expand_state(
        self,
        input_immutable_board,
        cllp_num_beams,
        cllp_num_return_sequences,
        return_raw_subgoals=False,
        **subgoal_generation_kwargs,
    ):
        subgoals = self.subgoal_generator.generate_subgoals(
            input_immutable_board, **subgoal_generation_kwargs
        )

        if return_raw_subgoals:
            return {subgoals[i]: None for i in range(len(subgoals))}


        paths = self.cllp.get_paths_batch(
            [(input_immutable_board, subgoal) for subgoal in subgoals], cllp_num_beams, cllp_num_return_sequences
        )


        subgoals_info = {}
        for subgoal, paths_to_subgoal in zip(subgoals, paths):
            subgoal_info = self.analyze_subgoal(input_immutable_board, subgoal, paths_to_subgoal)
            if subgoal_info is not None:
                subgoals_info[subgoal] = subgoal_info
        return subgoals_info

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
            "paths": correct_paths,
            "path_raw_probabilities": paths_raw_probabilities,
            "path_probabilities": paths_stats,
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
