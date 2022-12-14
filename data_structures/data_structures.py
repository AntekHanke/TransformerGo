from collections import namedtuple
from dataclasses import dataclass
from typing import Tuple

import chess

Transition = namedtuple("Transition", "immutable_board move move_number")
OneGameData = namedtuple("OneGameData", "metadata transitions")
LeelaSubgoal = namedtuple(
    "LeelaSubgoal",
    "input_idx target_idx input_immutable_board target_immutable_board dist_from_input input_level moves N Q D M P input_N",
)
# LeelaNodeData = namedtuple("LeelaNodeData", "id state moves_from_root level N Q D M P")

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