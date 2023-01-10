import chess
import numpy as np

from data_processing.data_utils import immutable_boards_to_img
from data_structures.data_structures import ImmutableBoard
from policy.chess_policy import LCZeroPolicy
from policy.cllp import CLLP
from subgoal_generator.subgoal_generator import BasicChessSubgoalGenerator
from utils.sample_fens import BLACK_MATE_IN_2
from value.chess_value import LCZeroValue


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
        n_subgoals,
        subgoal_generation_kwargs,
        cllp_num_beams,
        cllp_num_return_sequences,
    ):
        subgoals = self.subgoal_generator.generate_subgoals(input_immutable_board, n_subgoals, subgoal_generation_kwargs)
        paths = self.cllp.get_paths_batch(
            [(input_immutable_board, subgoal) for subgoal in subgoals], cllp_num_beams, cllp_num_return_sequences
        )

        # fig = immutable_boards_to_img([b] + subgoals, ["input"] + [f"subgoal{i}" for i in range(len(subgoals))])
        # fig.show()

        subgoals_info = {}

        for subgoal, paths_to_subgoal in zip(subgoals, paths):
            correct_paths = [path for path in paths_to_subgoal if verify_path(input_immutable_board, subgoal, path)]
            if len(correct_paths) == 0:
                continue

            paths_raw_probabilities = [
                self.policy.get_path_probabilities(input_immutable_board, path) for path in correct_paths
            ]

            paths_stats = [
                self.analyze_path_probabilities(path_raw_probabilities)
                for path_raw_probabilities in paths_raw_probabilities
            ]

            subgoal_info = {
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
                "highest_total_probability": max([path_stats["total_path_probability"] for path_stats in paths_stats]),
            }

            subgoals_info[subgoal] = subgoal_info

        return subgoals_info

    def analyze_path_probabilities(self, paths_raw_probabilities):
        total_path_probability = np.prod(paths_raw_probabilities)
        min_path_probability = min(paths_raw_probabilities)
        return {"total_path_probability": total_path_probability, "min_path_probability": min_path_probability}


generator = BasicChessSubgoalGenerator(
    "/home/tomasz/Research/subgoal_chess_data/local_leela_models/generator_athena/k=3"
)
cllp = CLLP("/home/tomasz/Research/subgoal_chess_data/local_leela_models/cllp/medium")
policy = LCZeroPolicy()
value = LCZeroValue()


expander = ChessStateExpander(policy, value, generator, cllp)

board = chess.Board()

# board = chess.Board(fen="r1b1kb1r/1pp4p/2p2p2/p3p3/4P1n1/2N2N2/PPP2PPP/R2Q2RK w Qkq - 10 31")
# board.push(chess.Move.from_uci("e2e4"))
# board.push(chess.Move.from_uci("e7e5"))
# board.push(chess.Move.from_uci("g1f3"))
# board.push(chess.Move.from_uci("b8c6"))
# board.push(chess.Move.from_uci("f1c4"))
# board.push(chess.Move.from_uci("g8f6"))
# board.push(chess.Move.from_uci("d2d4"))

b = ImmutableBoard.from_board(board)
subgoals_info = expander.expand_state(b, 12, 32, 16, 1)
subgoals = list(subgoals_info.keys())

prob_sub_list = []
for i, sub in enumerate(subgoals_info.values()):
    prob_sub_list.append(sub["highest_total_probability"])
    print(f"subgoal_{i}_prob = {sub['highest_total_probability']}")

print(f"max_prob = {max(prob_sub_list)}")

fig = immutable_boards_to_img([b] + subgoals, ["input"] + [f"s{i}:{[x.uci() for x in  subgoals_info[subgoals[i]]['paths'][0]]}" for i in range(len(subgoals))])
fig.show()
y = 4
