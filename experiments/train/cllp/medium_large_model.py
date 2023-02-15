import os
import sys

experiment_dir_path = os.path.dirname(os.path.abspath('__file__'))
sys.path.append(experiment_dir_path)

from datetime import date
from mrunner.helpers.specification_helper import create_experiments_helper
from experiments.train.cllp.common_training_params import medium_model_config

batch_size = {"ares": 1024, "athena": 350, "athena_2_gpu": 650, "athena_8_gpu": 2100}

MACHINE = "athena_8_gpu"

base_config = {
    "TrainModel.path_to_training_data": "/subgoal_chess_data/cllp_bigger",

    "GlobalParamsHandler.out_dir": f"/leela_models/cllp/medium_large_model/{date.today()}",
    "GlobalParamsHandler.path_type": "raw_path",

    "BartConfig.decoder_ffn_dim": 4096,
    "BartConfig.encoder_ffn_dim": 4096,
    "BartConfig.d_model": 1024,

    "TrainingArguments.per_device_train_batch_size": batch_size[MACHINE],
    "TrainingArguments.per_device_eval_batch_size": batch_size[MACHINE],
    "TrainingArguments.learning_rate": 1e-4,
}

params_grid = {"idx": [0]}

experiments_list = create_experiments_helper(
    experiment_name=f"b{batch_size[MACHINE]}-{base_config['TrainingArguments.learning_rate']}-lr-8g-medium-large--cllp",
    project_name="pmtest/subgoal-chess",
    base_config=dict(medium_model_config, **base_config),
    params_grid=params_grid,
    script="python3 -m runner --mrunner",
    exclude=["data", ".pytest_cache", "out", ".git"],
    python_path="",
    tags=["leela", "train", "medium-large", "cllp"],
    with_neptune=True,
    env={},
)
