import os
import sys

experiment_dir_path = os.path.dirname(os.path.abspath("__file__"))
sys.path.append(experiment_dir_path)

from datetime import date
from mrunner.helpers.specification_helper import create_experiments_helper

GEN_LONG_K_3 = "/home/tomasz/Research/subgoal_chess_data/local_leela_models/long_training/checkpoint-221500"
CLLP_PATH = "/home/tomasz/Research/subgoal_chess_data/local_leela_models/cllp/medium"
log_dir: str = "/home/tomasz/Research/subgoal_chess_data/bot_logs"

experiment_config = {
    "run.job_class": "@jobs.GameBetweenEngines",

    "GameBetweenEngines.generator_path": GEN_LONG_K_3,
    "GameBetweenEngines.cllp_path": CLLP_PATH,
    "GameBetweenEngines.out_dir": log_dir,

}

params_grid = {
    "idx": [0],
}

experiments_list = create_experiments_helper(
    experiment_name=f"generate_tree",
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
