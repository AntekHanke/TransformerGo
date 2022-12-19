import chess

from data_structures.data_structures import ImmutableBoard
from policy.chess_policy import BasicChessPolicy


def run_policy(checkpoint_path):
    policy = BasicChessPolicy(checkpoint_path)
    board = chess.Board()
    best_move = policy.get_best_moves(ImmutableBoard.from_board(board), 4)
    print(f"Best move: {best_move}")


run_policy("/home/tomasz/Research/subgoal_chess_data/local_leela_models/policy/ares_19.12")

