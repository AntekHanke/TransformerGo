from mrunner.helpers.specification_helper import create_experiments_helper

base_config = {"run.job_class": "@jobs.AnyJob"}

params_grid = {
    "idx": [0],
    "AnyJob.learning_rate": [1e-4],
    "AnyJob.k": [1,2,3,4,5][1:2]
}

experiments_list = create_experiments_helper(
    experiment_name="Train policy",
    project_name="pmtest/subgoal-chess",
    base_config=base_config,
    params_grid=params_grid,
    script="python3 -m runner --mrunner",
    exclude=["data", ".pytest_cache", "alpacka.egg-info", "out", ".git"],
    python_path="",
    tags=["train-policy"],
    with_neptune=True,
    env={},
)
