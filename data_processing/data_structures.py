from collections import namedtuple

import chess

ImmutableBoardData = namedtuple(
    "ImmutableBoard",
    "board active_player castles en_passant_target halfmove_clock fullmove_clock",
)


class ImmutableBoard(ImmutableBoardData):
    @classmethod
    def from_board(cls, board):
        fen_data = board.fen().split()
        return cls(*fen_data)

    def to_board(self):
        return chess.Board(fen=" ".join(self))
