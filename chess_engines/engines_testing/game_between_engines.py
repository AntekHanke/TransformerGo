import datetime
import os

import chess

from chess_engines.bots.basic_chess_engines import PolicyChess, RandomChessEngine
from chess_engines.bots.mcts_bot import MCTSChessEngine
from data_structures.data_structures import ImmutableBoard
from utils.data_utils import immutable_boards_to_img

GEN_LONG_K_3 = "/home/tomasz/Research/subgoal_chess_data/local_leela_models/long_training/checkpoint-221500"
CLLP_PATH = "/home/tomasz/Research/subgoal_chess_data/local_leela_models/cllp/medium"
log_dir: str = "/home/tomasz/Research/subgoal_chess_data/bot_logs"

board = chess.Board()
engine1 = MCTSChessEngine(
        time_limit=300,
        max_mcts_passes=20,
        generator_path=GEN_LONG_K_3,
        cllp_path=CLLP_PATH,
        cllp_num_beams=1,
        cllp_num_return_sequences=1,
        generator_num_beams=16,
        generator_num_subgoals=8,
        sort_subgoals_by="highest_total_probability",
        num_top_subgoals=6)
engine2 = PolicyChess(
    policy_checkpoint=None,
    log_dir=log_dir,
    debug_mode=True,
    replace_legall_move_with_random=False,
    do_sample=False,
    name="LeelaChessZero_POLICY",
    use_lczero_policy=True,
)

# engine1 = RandomChessEngine()
# engine2 = RandomChessEngine()

players = {'w': engine1, 'b': engine2}

c = 0
log_dir = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
os.makedirs(f"/home/tomasz/Research/subgoal_chess_data/games/{log_dir}")
while not board.is_game_over():

    if board.turn == chess.WHITE:
        engine = players['w']
        player = "MCTS"
    else:
        engine = players['b']
        player = "POLICY"

    move = engine.propose_best_moves(board, 1)
    print(f"Move {c} by {player}: {move}")
    board.push(chess.Move.from_uci(move))
    print("=====================================")
    print(board)
    fig = immutable_boards_to_img([ImmutableBoard.from_board(board)], [f"move {c}"])


    fig.savefig(f"/home/tomasz/Research/subgoal_chess_data/games/{log_dir}/move_{c}.png")
    c += 1

print(board.result())