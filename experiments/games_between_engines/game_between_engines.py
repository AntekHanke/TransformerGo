import os
import sys

experiment_dir_path = os.path.dirname(os.path.abspath("__file__"))
sys.path.append(experiment_dir_path)

from experiments.games_between_engines.engine_specifications import stockfish_engine_params, subgoal_mcts_engine_params
from mrunner.helpers.specification_helper import create_experiments_helper

GEN_LONG_K_3 = "/data_gg/out_models/medium_generator_from_scratch_with_all_subgoals/16_03_22/k_3/checkpoint-95000/"
CLLP_PATH = "/data_gg/out_models/medium_cllp_from_scratch_k_form_1_to_9/14_03_23/checkpoint-89500/"
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
