from collections import namedtuple

import chess

ImmutableBoardData = namedtuple(
    "ImmutableBoard",
    "board active_player castles en_passant_target halfmove_clock fullmove_clock",
)

SubgoalsFromModel = namedtuple("SubgoalFromModel", "input_immutable_board target_immutable_board probability")


class ImmutableBoard(ImmutableBoardData):
    @classmethod
    def from_board(cls, board):
        fen_data = board.fen().split()
        return cls(*fen_data)

    def to_board(self):
        return chess.Board(fen=" ".join(self))


# class SingleSubgoal:
#     def __init__(self, input_board, target_board):
#         # self.input_immutable_board = subgoal_from_model.input_immutable_board
#         # self.target_immutable_board = subgoal_from_model.target_immutable_board
#         # self.probability = subgoal_from_model.probability
#         #
#         self.input_value = None
#
#     def evaluate_with_stockfish(self):