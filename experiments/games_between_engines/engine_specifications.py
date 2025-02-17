from typing import Type

from chess_engines.bots.basic_chess_engines import PolicyChess, ChessEngine
from chess_engines.bots.mcts_bot import SubgoalMCTSChessEngine
from chess_engines.bots.stockfish_bot import StockfishBotEngine

stockfish_engine_params = {"stockfish_depth": 5, "stockfish_path": None}

subgoal_mcts_engine_params = {
    "time_limit": 300,
    "max_mcts_passes": 15,
    "generator_path": None,
    "cllp_path": None,
    "cllp_num_beams": 2,
    "cllp_num_return_sequences": 1,
    "generator_num_beams": 12,
    "generator_num_subgoals": 4,
    "subgoal_distance_k": 3,
    "sort_subgoals_by": "highest_min_probability",
    "subgoal_probs_opponent_only": True,
    "num_top_subgoals": 4,
    "first_n_moves_sampled": 2,
    "log_trees": False,
    "log_dir": None,
}

vanilla_mcts_engine_params = {
    "time_limit": 300,
    "max_mcts_passes": 15,
    "policy_num_moves": 4,
    "policy_num_beams": 12,
    "first_n_moves_sampled": 2,
    "log_trees": False,
    "log_dir": None,
}

leela_engine_params = {
    "policy_checkpoint": None,
    "log_dir": None,
    "debug_mode": True,
    "replace_legall_move_with_random": True,
    "do_sample": True,
    "name": "LeelaChessZero_POLICY",
    "use_lczero_policy": True,
}
