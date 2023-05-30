import os
import sys

experiment_dir_path = os.path.dirname(os.path.abspath("__file__"))
sys.path.append(experiment_dir_path)

from experiments.data_loading_params import policy_data_loading_header, ultra_small_data
from experiments.models_params import common_train_params, AlphaZeroModel
from mrunner.helpers.specification_helper import create_experiments_helper

TRAIN_TYPE = "from_scratch_go_policy"
HISTORY = ""
OUT_DIR = "/godata"

experiment_config = dict(
    **common_train_params[TRAIN_TYPE],
    **policy_data_loading_header[TRAIN_TYPE + HISTORY],
    **ultra_small_data[TRAIN_TYPE],
    **AlphaZeroModel,
)
experiment_config["TrainConvolutionFromScratch.out_dir"] = OUT_DIR

params_grid = {
    "idx": [0],
}

experiments_list = create_experiments_helper(
    experiment_name=f"go_conv_policy_train_with_history",
    project_name="pmtest/subgoal-chess",
    base_config=experiment_config,
    params_grid=params_grid,
    script="python3 -m runner --mrunner",
    exclude=["model_checkpoints", "data", ".pytest_cache", "out", ".git", "exclude", "lib", "lib64"],
    python_path="",
    tags=["train", "policy", "go", "convolution", "from_scratch", "with_history"],
    with_neptune=True,
    env={},
)
