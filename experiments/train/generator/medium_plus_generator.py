import os
import sys

experiment_dir_path = os.path.dirname(os.path.abspath('__file__'))
sys.path.append(experiment_dir_path)

from datetime import date
from mrunner.helpers.specification_helper import create_experiments_helper
from experiments.train.generator.common_training_params import medium_model_config, generator_global_params

base_config = {
    "GlobalParamsHandler.out_dir": f"/models/generator/medium_model/{date.today()}",

    "BartConfig.encoder_layers": 10,
    "BartConfig.decoder_layers": 10,
    "BartConfig.encoder_attention_heads": 16,
    "BartConfig.decoder_attention_heads": 16,

    "TrainingArguments.per_device_train_batch_size": 2100,
    "TrainingArguments.per_device_eval_batch_size": 2000,
}

params_grid = {
    # "GlobalParamsHandler.k": [1, 2, 3, 4, 5, 6],
    "GlobalParamsHandler.learning_rate": [3e-5],
}

batch = base_config["TrainingArguments.per_device_train_batch_size"]

experiments_list = create_experiments_helper(
    experiment_name=f"-training-medium-plus-generator_b={batch}",
    project_name="pmtest/subgoal-chess",
    base_config=dict(medium_model_config, **generator_global_params, **base_config),
    params_grid=params_grid,
    script="python3 -m runner --mrunner",
    exclude=["data", ".pytest_cache", "out", ".git"],
    python_path="",
    tags=["leela", "train", "medium-plus", "subgoals", "batch"],
    with_neptune=True,
    env={},
)
