import time

import chess

from data_structures.data_structures import ImmutableBoard
from policy.chess_policy import LCZeroPolicy


class DebugJob:
    """Any code goes here"""
    def execute(self):
        policy = LCZeroPolicy()
        board = chess.Board()
        time_start = time.time()
        for _ in range(30):
            moves = policy.get_best_moves(ImmutableBoard.from_board(board), 5)
            print(moves)
            board.push(moves[0])
        print(time.time() - time_start)