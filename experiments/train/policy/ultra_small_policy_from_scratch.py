import os
import sys

experiment_dir_path = os.path.dirname(os.path.abspath("__file__"))
sys.path.append(experiment_dir_path)

from experiments.data_loading_params import policy_data_loading_header, ultra_small_data
from experiments.models_params import common_train_params, ultra_small_model
from mrunner.helpers.specification_helper import create_experiments_helper

TRAIN_TYPE = "from_scratch"
HISTORY = ""
OUT_DIR = "/out_models/ultra_small_policy_from_scratch"

experiment_config = dict(
    **common_train_params[TRAIN_TYPE],
    **policy_data_loading_header[TRAIN_TYPE + HISTORY],
    **ultra_small_data[TRAIN_TYPE],
    **ultra_small_model,
)
experiment_config["TrainModelFromScratch.out_dir"] = OUT_DIR

params_grid = {
    "idx": [0],
}

experiments_list = create_experiments_helper(
    experiment_name=f"ultra_small_policy_train",
    project_name="pmtest/subgoal-chess",
    base_config=experiment_config,
    params_grid=params_grid,
    script="python3 -m runner --mrunner",
    exclude=["data", ".pytest_cache", "out", ".git"],
    python_path="",
    tags=["train", "small", "policy", "from_scratch"],
    with_neptune=True,
    env={},
)
