import chess

from data_processing.data_utils import immutable_boards_to_img
from data_structures.data_structures import ImmutableBoard
from policy.cllp import CLLP
from subgoal_generator.subgoal_generator import BasicChessSubgoalGenerator


def verify_path(input_immutable_board, subgoal, path):
    # try:
    board = input_immutable_board.to_board()
    for move in path:
        board.push(chess.Move.from_uci(move))
        if ImmutableBoard.from_board(board) == subgoal:
            return True
    # except Exception as e:
    #     print(e)
    #     return False


class ChessStateExpander:
    def __init__(self, subgoal_generator, cllp):
        # self.policy = policy
        # self.value = value
        self.subgoal_generator = subgoal_generator
        self.cllp = cllp

    def expand_state(self, input_immutable_board, n_subgoals, num_beams, num_return_sequences):
        subgoals = self.subgoal_generator.generate_subgoals(input_immutable_board, n_subgoals)
        # subgoal_values = [self.value.evaluate_immutable_board(subgoal) for subgoal in subgoals]
        paths = self.cllp.get_paths_batch(
            [(input_immutable_board, subgoal) for subgoal in subgoals], num_beams, num_return_sequences
        )

        subgoals_info = {subgoal: {} for subgoal in subgoals}

        for subgoal, paths_to_subgoal in zip(subgoals, paths):
            correct_paths = [path for path in paths_to_subgoal if verify_path(input_immutable_board, subgoal, path)]
            incorrect_paths = [path for path in paths_to_subgoal if not verify_path(input_immutable_board, subgoal, path)]

            print(f"CORRECT: {correct_paths}")
            print(f"INCORRECT: {incorrect_paths}")

            subgoals_info[subgoal]["correct_paths"] = correct_paths

        return subgoals, subgoals_info


generator = BasicChessSubgoalGenerator(
    "/home/tomasz/Research/subgoal_chess_data/local_leela_models/4gpu_generator/subgoals_k=3"
)
cllp = CLLP("/home/tomasz/Research/subgoal_chess_data/local_leela_models/cllp/medium")

expander = ChessStateExpander(generator, cllp)

board = chess.Board()
board.push(chess.Move.from_uci("e2e4"))
board.push(chess.Move.from_uci("e7e5"))
board.push(chess.Move.from_uci("g1f3"))

b = ImmutableBoard.from_board(board)
subgoals, result = expander.expand_state(b, 4, 16, 2)
fig = immutable_boards_to_img([b] + subgoals, ["input"] + [f"subgoal {i}" for i in range(len(subgoals))])
fig.show()
print(result)

