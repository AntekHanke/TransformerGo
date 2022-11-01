import chess

from data_structures.data_structures import ImmutableBoard
from policy.cllp import CLLP

# x = CLLP("/home/tomasz/Research/subgoal_chess_data/local_leela_models/cllp_one_move/final_model")
y = CLLP("/home/tomasz/Research/subgoal_chess_data/local_leela_models/cllp_all_moves/final_model")

board = chess.Board()
b1 = ImmutableBoard.from_board(board)
board.push(chess.Move.from_uci("e2e4"))
board.push(chess.Move.from_uci("e7e5"))
board.push(chess.Move.from_uci("g1f3"))
b2 = ImmutableBoard.from_board(board)
moves = y.get_path(b1, b2)
print(moves)
