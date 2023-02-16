import os
import sys

experiment_dir_path = os.path.dirname(os.path.abspath("__file__"))
sys.path.append(experiment_dir_path)

from mrunner.helpers.specification_helper import create_experiments_helper


experiment_config = {
    "TrainModelFromScratch.out_dir": "/out_models/ultra_small_generator_from_scratch",
}

base_config = dict(ultra_small_model_config, **experiment_config, **generator_global_params, **ultra_small_train_header)

params_grid = {
    "idx": [0],
}

experiments_list = create_experiments_helper(
    experiment_name=f"ultra_small_generator_from_scratch",
    project_name="pmtest/subgoal-chess",
    base_config=base_config,
    params_grid=params_grid,
    script="python3 -m runner --mrunner",
    exclude=["data", ".pytest_cache", "out", ".git"],
    python_path="",
    tags=["train", "small", "subgoals", "from_scratch"],
    with_neptune=True,
    env={},
)
