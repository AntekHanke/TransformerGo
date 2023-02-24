import os
import sys

experiment_dir_path = os.path.dirname(os.path.abspath("__file__"))
sys.path.append(experiment_dir_path)

from experiments.data_loading_params import subgoal_data_loading_header, ultra_small_data
from experiments.models_params import common_train_params
from mrunner.helpers.specification_helper import create_experiments_helper

TRAIN_TYPE = "resume"
OUT_DIR = "/out_models/ultra_small_generator_resume"

experiment_config = dict(
    **common_train_params[TRAIN_TYPE], **subgoal_data_loading_header[TRAIN_TYPE], **ultra_small_data[TRAIN_TYPE]
)
experiment_config["ResumeTraining.out_dir"] = OUT_DIR

experiment_config["ResumeTraining.checkpoint_path"] = "/out_models/ultra_small_generator_from_scratch/"
experiment_config["ResumeTraining.checkpoint_num"] = 500

params_grid = {
    "idx": [0],
}

experiments_list = create_experiments_helper(
    experiment_name=f"ultra_small_generator_resume",
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
