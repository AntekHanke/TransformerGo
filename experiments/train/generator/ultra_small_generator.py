import random
from datetime import date
from mrunner.helpers.specification_helper import create_experiments_helper
from common_training_params import ultra_small_model_config, generator_global_params

base_config = {
    "TrainModel.files_batch_size": 5,
    "GlobalParamsHandler.out_dir": f"/out/ultra_small_generator/{date.today()}/{random.randint(0, 1000000)}/",
    "GlobalParamsHandler.path_format": ["k"],
}

params_grid = {
    "idx": [0],
    "GlobalParamsHandler.k": [3],
    "GlobalParamsHandler.learning_rate": [3e-5],
}

experiments_list = create_experiments_helper(
    experiment_name=f"ultra_small_generator-{base_config['GlobalParamsHandler.out_dir']}",
    project_name="pmtest/subgoal-chess",
    base_config=dict(ultra_small_model_config, **generator_global_params, **base_config),
    params_grid=params_grid,
    script="python3 -m runner --mrunner",
    exclude=["data", ".pytest_cache", "out", ".git"],
    python_path="",
    tags=["chessengines games", "train", "small", "subgoals", "k=3"],
    with_neptune=True,
    env={},
)
