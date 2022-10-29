import chess

from data_structures.data_structures import ImmutableBoard
from policy.chess_policy import BasicChessPolicy

policy = BasicChessPolicy("/home/tomasz/Research/subgoal_chess_data/local_leela_models/policy/final_model")

board = chess.Board()
move = policy.get_best_move(ImmutableBoard.from_board(board))
print(move.uci())
