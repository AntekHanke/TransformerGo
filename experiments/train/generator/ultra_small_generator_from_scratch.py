import os
import sys

from experiments.train.generator.data_loading_params import data_loading_header, ultra_small_data

experiment_dir_path = os.path.dirname(os.path.abspath("__file__"))
sys.path.append(experiment_dir_path)

from experiments.train.generator.models_params import common_train_params, ultra_small_model
from mrunner.helpers.specification_helper import create_experiments_helper

TRAIN_TYPE = "from_scratch"
OUT_DIR = "/out_models/ultra_small_generator_from_scratch"

experiment_config = dict(
    **common_train_params[TRAIN_TYPE], **data_loading_header, **ultra_small_data[TRAIN_TYPE], **ultra_small_model
)
experiment_config["TrainModelFromScratch.out_dir"] = OUT_DIR

params_grid = {
    "idx": [0],
}

experiments_list = create_experiments_helper(
    experiment_name=f"ultra_small_generator_train",
    project_name="pmtest/subgoal-chess",
    base_config=experiment_config,
    params_grid=params_grid,
    script="python3 -m runner --mrunner",
    exclude=["data", ".pytest_cache", "out", ".git"],
    python_path="",
    tags=["train", "small", "subgoals", "from_scratch"],
    with_neptune=True,
    env={},
)
