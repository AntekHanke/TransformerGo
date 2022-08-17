from collections import namedtuple
from dataclasses import dataclass
from typing import Optional

import chess

Transition = namedtuple("Transition", "immutable_board move")
OneGameData = namedtuple("OneGameData", "metadata, transitions")

ImmutableBoardData = namedtuple(
    "ImmutableBoard",
    "board active_player castles en_passant_target halfmove_clock fullmove_clock",
)

SubgoalsFromModel = namedtuple("SubgoalFromModel", "input_immutable_board target_immutable_board")

@dataclass
class ChessMetadata:
    """Stores arbitrary metadata about a single game. Different games have different metadata fields."""
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

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
