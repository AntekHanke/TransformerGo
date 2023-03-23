from typing import Type

from chess_engines.bots.basic_chess_engines import PolicyChess, ChessEngine
from chess_engines.bots.mcts_bot import SubgoalMCTSChessEngine
from chess_engines.bots.stockfish_bot import StockfishBotEngine


class EngineParameters:
    def __init__(self, engine_class: Type[ChessEngine], engine_params: dict):
        self.engine_class = engine_class
        self.engine_params = engine_params


stockfish_engine_spec = EngineParameters(StockfishBotEngine, {"stockfish_depth": 5, "stockfish_path": None})

subgoal_mcts_engine_spec = EngineParameters(
    SubgoalMCTSChessEngine,
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
)

leela_engine_spec = EngineParameters(
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
)
