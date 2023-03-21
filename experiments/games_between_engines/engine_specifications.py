from chess_engines.bots.basic_chess_engines import PolicyChess
from chess_engines.bots.mcts_bot import MCTSChessEngine
from chess_engines.bots.stockfish_bot import StockfishBotEngine

stockfish = [
    StockfishBotEngine,
    {
        "stockfish_depth": 5,
        "stockfish_path": None,
    },
]

mcts_small = [
    MCTSChessEngine,
    {
        "time_limit": 300,
        "max_mcts_passes": 15,
        "generator_path": None,
        "cllp_path": None,
        "cllp_num_beams": 1,
        "cllp_num_return_sequences": 1,
        "generator_num_beams": 8,
        "generator_num_subgoals": 4,
        "sort_subgoals_by": None,
        "num_top_subgoals": 4,
    },
]

leela = [
    PolicyChess,
    {
        "policy_checkpoint": None,
        "log_dir": None,
        "debug_mode": True,
        "replace_legall_move_with_random": False,
        "do_sample": False,
        "name": "LeelaChessZero_POLICY",
        "use_lczero_policy": True,
    },
]
