from collections import namedtuple
from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Tuple

import chess

Transition = namedtuple("Transition", "immutable_board move move_number")
OneGameData = namedtuple("OneGameData", "metadata transitions")
LeelaSubgoal = namedtuple(
    "LeelaSubgoal",
    "input_idx target_idx input_immutable_board target_immutable_board dist_from_input input_level moves N Q D M P input_N",
)

SubgoalsFromModel = namedtuple("SubgoalFromModel", "input_immutable_board target_immutable_board")


class HistoryLength(IntEnum):
    no_history = 0
    short_history = 10


@dataclass
class ChessMetadata:
    """Stores arbitrary metadata about a single game. Different games have different metadata fields."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


ImmutableBoardData = namedtuple(
    "ImmutableBoard",
    "board active_player castles en_passant_target halfmove_clock fullmove_clock",
)


class ImmutableBoard(ImmutableBoardData):
    @classmethod
    def from_board(cls, board: chess.Board) -> "ImmutableBoard":
        fen_data = board.fen().split()
        return cls(*fen_data)

    @classmethod
    def from_fen_str(cls, fen: str) -> "ImmutableBoard":
        return ImmutableBoard(*fen.split())

    def to_board(self) -> chess.Board:
        return chess.Board(fen=" ".join(self))

    def act(self, move: chess.Move) -> "ImmutableBoard":
        chess_board = self.to_board()
        chess_board.push(move)
        return ImmutableBoard.from_board(chess_board)

    def legal_moves(self) -> Tuple[chess.Move]:
        return tuple(self.to_board().legal_moves)

    def fen(self) -> str:
        return " ".join(self)

    def __hash__(self):
        return hash(self.board + self.active_player + self.castles + self.en_passant_target)

    def __eq__(self, other):
        return all(
            [
                self.board == other.board,
                self.active_player == other.active_player,
                self.castles == other.castles,
                self.en_passant_target == other.en_passant_target,
            ]
        )
