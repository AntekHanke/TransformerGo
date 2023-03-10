import time

import chess

from data_structures.data_structures import ImmutableBoard
from policy.chess_policy import LCZeroPolicy


class DebugJob:
    """Any code goes here"""

    def execute(self):
        import chess
        from data_structures.data_structures import ImmutableBoard
        from policy.chess_policy import LCZeroPolicy

        policy = LCZeroPolicy()
        b = chess.Board()
        bb = ImmutableBoard.from_board(b)
        y = policy.get_best_moves(bb)
        print(y)
