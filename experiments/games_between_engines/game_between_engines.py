import os
import sys

experiment_dir_path = os.path.dirname(os.path.abspath("__file__"))
sys.path.append(experiment_dir_path)

from experiments.games_between_engines.engine_specifications import stockfish_engine_params, subgoal_mcts_engine_params
from mrunner.helpers.specification_helper import create_experiments_helper

GEN_LONG_K_3 = "/data_mg/generators/generator_k_3/checkpoint-221500/"
CLLP_PATH = "/data_mg/cllp/medium/"
STOCKFISH_PATH = "/data_mg/stockfish/stockfish_15_linux_x64/stockfish_15_x64"
log_dir: str = "/out_models/bot_logs"

white_engine_params = subgoal_mcts_engine_params
white_engine_params["generator_path"] = GEN_LONG_K_3
white_engine_params["cllp_path"] = CLLP_PATH
white_engine_params["sort_subgoals_by"] = "highest_min_probability"

black_engine_params = stockfish_engine_params
black_engine_params["stockfish_path"] = STOCKFISH_PATH

experiment_config = {
    "run.job_class": "@jobs.GameBetweenEngines",

    "GameBetweenEngines.engine_white_class": "@chess_engines.SubgoalMCTSChessEngine",
    "GameBetweenEngines.engine_black_class": "@chess_engines.StockfishBotEngine",
    "GameBetweenEngines.engine_white_params": white_engine_params,
    "GameBetweenEngines.engine_black_params": black_engine_params,
    "GameBetweenEngines.eval_stockfish_path": STOCKFISH_PATH,
    "GameBetweenEngines.eval_stockfish_depth": 20,
    "GameBetweenEngines.out_dir": log_dir,
    "GameBetweenEngines.debug_mode": False,

    "use_neptune": True
}

params_grid = {
    "idx": [0],
}

experiments_list = create_experiments_helper(
    experiment_name=f"game",
    project_name="pmtest/subgoal-chess",
    base_config=experiment_config,
    params_grid=params_grid,
    script="python3 -m runner --mrunner",
    exclude=["data", ".pytest_cache", "out", ".git"],
    python_path="",
    tags=["generate", "tree", "mcts"],
    with_neptune=True,
    env={},
)
