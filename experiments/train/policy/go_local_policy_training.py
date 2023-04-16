import os
import sys

experiment_dir_path = os.path.dirname(os.path.abspath("__file__"))
sys.path.append(experiment_dir_path)

from experiments.data_loading_params import policy_data_loading_headerGo, go_ultra_small_data
from experiments.models_params import common_train_params, go_ultra_small_model
from mrunner.helpers.specification_helper import create_experiments_helper
from local_runner import local_run

TRAIN_TYPE = "from_scratch"
HISTORY = ""
OUT_DIR = "/plgantekhanke/models/ultra_small_policy_from_scratch"

experiment_config = dict(
    **common_train_params[TRAIN_TYPE],
    **policy_data_loading_headerGo[TRAIN_TYPE + HISTORY],
    **go_ultra_small_data[TRAIN_TYPE],
    **go_ultra_small_model,
)
experiment_config["TrainModelFromScratch.out_dir"] = OUT_DIR

experiment_config["TrainModelFromScratch.path_to_training_data"] = "/plgantekhanke/sgfs/tokenizeddata/trainval_train_part_0.pkl",
experiment_config["TrainModelFromScratch.path_to_eval_data"] = "/plgantekhanke/sgfs/tokenizeddata/evalval_eval_part_0.pkl",

params_grid = {
    "idx": [0],
}

experiments_list = create_experiments_helper(
    experiment_name=f"ultra_small_policy_train",
    project_name="pmtest/subgoal-chess",
    base_config=experiment_config,
    params_grid=params_grid,
    script="python3 -m runner --mrunner",
    exclude=["data", ".pytest_cache", "out", ".git", "exclude"],
    python_path="",
    tags=["train", "small", "policy", "from_scratch"],
    with_neptune=True,
    env={},
)


