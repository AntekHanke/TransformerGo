from datetime import date
from mrunner.helpers.specification_helper import create_experiments_helper
from common_training_params import medium_model_config, generator_global_params

base_config = {
    "GlobalParamsHandler.out_dir": f"/models/generator/medium_model/{date.today()}",
}

params_grid = {
    # "GlobalParamsHandler.k": [1, 2, 3, 4, 5, 6],
    "GlobalParamsHandler.learning_rate": [3e-5],
}

batch = base_config["TrainingArguments.per_device_train_batch_size"]

experiments_list = create_experiments_helper(
    experiment_name=f"-training-medium-generator_b={batch}",
    project_name="pmtest/subgoal-chess",
    base_config=dict(medium_model_config, **generator_global_params, **base_config),
    params_grid=params_grid,
    script="python3 -m runner --mrunner",
    exclude=["data", ".pytest_cache", "out", ".git"],
    python_path="",
    tags=["leela", "train", "medium", "subgoals", "batch"],
    with_neptune=True,
    env={},
)
