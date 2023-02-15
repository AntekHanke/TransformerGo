import os
import sys

experiment_dir_path = os.path.dirname(os.path.abspath('__file__'))
sys.path.append(experiment_dir_path)

import random
from datetime import date
from mrunner.helpers.specification_helper import create_experiments_helper
from experiments.train.cllp.common_training_params import medium_model_config

batch_size = {"ares": 1024, "athena": 700, "athena_2_gpu": 1350, "athena_4_gpu": 2700}

MACHINE = "athena_4_gpu"

base_config = {
    "TrainModel.path_to_training_data": "/subgoal_chess_data/cllp_data",

    "GlobalParamsHandler.out_dir": f"/leela_models/cllp/medium_model/{date.today()}/{random.randint(0, 100000)}",
    # "GlobalParamsHandler.path_format": ['learning_rate'],

    "TrainingArguments.per_device_train_batch_size": batch_size[MACHINE],
    "TrainingArguments.per_device_eval_batch_size": batch_size[MACHINE],
}

params_grid = {
    "idx": [0],
    # "GlobalParamsHandler.learning_rate": [3e-5, 3e-4]
}

experiments_list = create_experiments_helper(
    experiment_name=f"EQ-b{batch_size[MACHINE]}-{base_config['TrainingArguments.learning_rate']}-lr-2g-medium-cllp",
    project_name="pmtest/subgoal-chess",
    base_config=dict(medium_model_config, **base_config),
    params_grid=params_grid,
    script="python3 -m runner --mrunner",
    exclude=["data", ".pytest_cache", "out", ".git"],
    python_path="",
    tags=["leela", "train", "medium", "cllp"],
    with_neptune=True,
    env={},
)
