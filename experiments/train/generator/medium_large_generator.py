import os
import sys

experiment_dir_path = os.path.dirname(os.path.abspath('__file__'))
sys.path.append(experiment_dir_path)

from datetime import date
from mrunner.helpers.specification_helper import create_experiments_helper
from experiments.train.generator.common_training_params import medium_model_config, generator_global_params

base_config = {
    "TrainModel.path_to_training_data": None,
    "TrainModel.path_to_eval_data": None,

    "GlobalParamsHandler.k": 1,
    "GlobalParamsHandler.data_location": "/chess_data",
    "GlobalParamsHandler.out_dir": f"/leela_models/generator/large_model/{date.today()}",

    "BartConfig.decoder_ffn_dim": 4096,
    "BartConfig.encoder_ffn_dim": 4096,
    "BartConfig.d_model": 1024,

    "TrainingArguments.per_device_train_batch_size": 2300,
    "TrainingArguments.per_device_eval_batch_size": 2300,
    "TrainingArguments.learning_rate": 1e-4,
}

params_grid = {
    "idx": [0],
    "GlobalParamsHandler.k": [3, 4, 5, 1, 2, 6],
    "GlobalParamsHandler.learning_rate": [1e-4],
}

experiments_list = create_experiments_helper(
    experiment_name=f"-training-medium_large-generator-leela-subgoals",
    project_name="pmtest/subgoal-chess",
    base_config=dict(medium_model_config, **generator_global_params, **base_config),
    params_grid=params_grid,
    script="python3 -m runner --mrunner",
    exclude=["data", ".pytest_cache", "out", ".git"],
    python_path="",
    tags=["leela", "train", "subgoals", "medium_large", "all_k"],
    with_neptune=True,
    env={},
)
