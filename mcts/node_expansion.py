import chess
import numpy as np

from data_processing.data_utils import immutable_boards_to_img
from data_structures.data_structures import ImmutableBoard
from policy.cllp import CLLP
from subgoal_generator.subgoal_generator import BasicChessSubgoalGenerator
from value.chess_value import LCZeroValue


def verify_path(input_immutable_board, subgoal, path):
    try:
        board = input_immutable_board.to_board()
        for move in path:
            board.push(chess.Move.from_uci(move))
            if ImmutableBoard.from_board(board) == subgoal:
                return True
        return False
    except:
        return False


class ChessStateExpander:
    def __init__(self, chess_policy, chess_value, subgoal_generator, cllp):
        self.policy = chess_policy
        self.value = chess_value
        self.subgoal_generator = subgoal_generator
        self.cllp = cllp

    def expand_state(
        self,
        input_immutable_board,
        n_subgoals,
        cllp_num_beams,
        cllp_num_return_sequences,
        return_only_correct_subgoals=True,
    ):
        subgoals = self.subgoal_generator.generate_subgoals(input_immutable_board, n_subgoals)
        paths = self.cllp.get_paths_batch(
            [(input_immutable_board, subgoal) for subgoal in subgoals], cllp_num_beams, cllp_num_return_sequences
        )

        subgoals_info = {}

        for subgoal, paths_to_subgoal in zip(subgoals, paths):
            correct_paths = [path for path in paths_to_subgoal if verify_path(input_immutable_board, subgoal, path)]
            if return_only_correct_subgoals and len(correct_paths) == 0:
                continue

            paths_raw_probabilities = [
                self.policy.get_path_probabilities(input_immutable_board, subgoal, path) for path in paths_to_subgoal
            ]

            paths_stats = [
                self.analyze_path_probabilities(path_raw_probabilities)
                for path_raw_probabilities in paths_raw_probabilities
            ]

            subgoal_info = {
                "value": self.value.evaluate_immutable_board(subgoal),
                "paths": paths_to_subgoal,
                "path_probabilities": paths_stats,
                "path_with_highest_min_probability": paths_to_subgoal[
                    np.argmax([path_stats["min_path_probability"] for path_stats in paths_stats])
                ],
                "path_with_highest_total_probability": paths_to_subgoal[
                    np.argmax([path_stats["total_path_probability"] for path_stats in paths_stats])
                ],
                "highest_min_probability": max([path_stats["min_path_probability"] for path_stats in paths_stats]),
                "highest_total_probability": max([path_stats["total_path_probability"] for path_stats in paths_stats]),
            }

            subgoals_info[subgoal] = subgoal_info

        return subgoals, subgoals_info

    def analyze_path_probabilities(self, paths_raw_probabilities):
        total_path_probability = np.prod(paths_raw_probabilities)
        min_path_probability = min(paths_raw_probabilities)
        return {"total_path_probability": total_path_probability, "min_path_probability": min_path_probability}


generator = BasicChessSubgoalGenerator(
    "/home/tomasz/Research/subgoal_chess_data/local_leela_models/4gpu_generator/subgoals_k=3"
)
cllp = CLLP("/home/tomasz/Research/subgoal_chess_data/local_leela_models/cllp/medium")
value = LCZeroValue()

expander = ChessStateExpander(value, generator, cllp)

board = chess.Board()
board.push(chess.Move.from_uci("e2e4"))
board.push(chess.Move.from_uci("e7e5"))
board.push(chess.Move.from_uci("g1f3"))

b = ImmutableBoard.from_board(board)
subgoals, result, vals = expander.expand_state(b, 4, 16, 2)
fig = immutable_boards_to_img([b] + subgoals, vals)
fig.show()
print(result)
