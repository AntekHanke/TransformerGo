import os
import sys

experiment_dir_path = os.path.dirname(os.path.abspath('__file__'))
sys.path.append(experiment_dir_path)

from datetime import date
from mrunner.helpers.specification_helper import create_experiments_helper
from experiments.train.generator.common_training_params import small_model_config, generator_global_params

base_config = {
    "TrainModel.files_batch_size": 10,

    "GlobalParamsHandler.out_dir": f"/leela_models/generator/small_model/{date.today()}/",

    "BartConfig.encoder_layers": 6,
    "BartConfig.decoder_layers": 6,
    "BartConfig.encoder_attention_heads": 4,
    "BartConfig.decoder_attention_heads": 4,
    "BartConfig.decoder_ffn_dim": 512,
    "BartConfig.encoder_ffn_dim": 512,
    "BartConfig.d_model": 128,
}

params_grid = {
    "idx": [0],
    "GlobalParamsHandler.k": [3],
    "GlobalParamsHandler.learning_rate": [3e-5],
}

experiments_list = create_experiments_helper(
    experiment_name=f"-training-small-generator-k3-chessdata-chessengines",
    project_name="pmtest/subgoal-chess",
    base_config=dict(small_model_config, **generator_global_params, **base_config),
    params_grid=params_grid,
    script="python3 -m runner --mrunner",
    exclude=["data", ".pytest_cache", "out", ".git"],
    python_path="",
    tags=["chessengines games", "train", "small", "subgoals", "k=3"],
    with_neptune=True,
    env={},
)
