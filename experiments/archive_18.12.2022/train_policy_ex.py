from mrunner.helpers.specification_helper import create_experiments_helper

base_config = {"run.job_class": "@jobs.AnyJob"}

params_grid = {
    "idx": [0],
    "AnyJob.learning_rate": [1e-6, 3e-6, 5e-6, 1e-5, 3e-5, 5e-5, 1e-4],
}

experiments_list = create_experiments_helper(
    experiment_name="Train pgn_policy",
    project_name="pmtest/subgoal-chess",
    base_config=base_config,
    params_grid=params_grid,
    script="python3 -m runner --mrunner",
    exclude=["data", ".pytest_cache", "alpacka.egg-info", "out", ".git"],
    python_path="",
    tags=["train-pgn_policy"],
    with_neptune=True,
    env={},
)
