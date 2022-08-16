from collections import namedtuple
from typing import Optional

import chess

ImmutableBoardData = namedtuple(
    "ImmutableBoard",
    "board active_player castles en_passant_target halfmove_clock fullmove_clock",
)

SubgoalsFromModel = namedtuple("SubgoalFromModel", "input_immutable_board target_immutable_board")


class ImmutableBoard(ImmutableBoardData):
    @classmethod
    def from_board(cls, board):
        fen_data = board.fen().split()
        return cls(*fen_data)

    def to_board(self):
        return chess.Board(fen=" ".join(self))


class SingleSubgoal:
    def __init__(
        self,
        input_immutable_board,
        target_immutable_board,
        input_value: Optional[float] = None,
        target_value: Optional[float] = None,
    ):
        self.input_immutable_board = input_immutable_board
        self.target_immutable_board = target_immutable_board
        self.input_value = input_value
        self.target_value = target_value

        if self.input_value is not None and self.target_value is not None:
            self.delta = self.target_value - self.input_value

    def evaluate_with_stockfish(self):
        pass
