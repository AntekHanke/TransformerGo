from mrunner.helpers.specification_helper import create_experiments_helper

base_config = {"run.job_class": "@jobs.TrainModel"}

params_grid = {
    "idx": [0],
}

experiments_list = create_experiments_helper(
    experiment_name="Quality data",
    project_name="pmtest/subgoal-chess",
    base_config=base_config,
    params_grid=params_grid,
    script="python3 -m runner --mrunner",
    exclude=["data", ".pytest_cache", "alpacka.egg-info", "out", ".git"],
    python_path="",
    tags=["quality-data"],
    with_neptune=True,
    env={},
)
