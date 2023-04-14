import os
import sys

experiment_dir_path = os.path.dirname(os.path.abspath("__file__"))
sys.path.append(experiment_dir_path)

from experiments.games_between_engines.engine_specifications import subgoal_mcts_engine_params
from mrunner.helpers.specification_helper import create_experiments_helper
from configures.global_config import STOCKFISH_PATH

GEN_LONG_K_3 = "/data_gg/out_models/medium_generator_from_scratch_with_all_subgoals/16_03_22/k_3/checkpoint-280500/"
CLLP_PATH = "/data_gg/out_models/medium_cllp_from_scratch_k_form_1_to_9/14_03_23/checkpoint-144500/"

white_engine_params = subgoal_mcts_engine_params.copy()
white_engine_params["generator_path"] = GEN_LONG_K_3
white_engine_params["cllp_path"] = CLLP_PATH
white_engine_params["max_mcts_passes"] = 15

black_engine_params = subgoal_mcts_engine_params.copy()
black_engine_params["generator_path"] = GEN_LONG_K_3
black_engine_params["cllp_path"] = CLLP_PATH
black_engine_params["max_mcts_passes"] = 15

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
    experiment_name=f"game-white-subgoal-mcts-?-black-subgoal-mcts-?",
    project_name="pmtest/subgoal-chess",
    base_config=experiment_config,
    params_grid=params_grid,
    script="python3 -m runner --mrunner",
    exclude=["data", ".pytest_cache", "out", ".git"],
    python_path="",
    tags=["mcts", "game", "self-play", "mcts"],
    with_neptune=True,
    env={},
)
