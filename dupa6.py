import chess

from chess_engines.bots.basic_chess_engines import SubgoalWithCLLPStockfish

engine = SubgoalWithCLLPStockfish(
    "/home/tomasz/Research/subgoal_chess_data/local_leela_models/generator/medium_k=1/final_model",
    "/home/tomasz/Research/subgoal_chess_data/local_leela_models/cllp_all_moves/final_model",
    6
)

board = chess.Board()

move = engine.propose_best_move(board)
print(move)
