from mrunner.helpers.specification_helper import create_experiments_helper

base_config = {"run.job_class": "@jobs.AnyJob", "use_neptune": True}

params_grid = {
    "idx": [0],
    "AnyJob.learning_rate": [3e-4],
    "AnyJob.k": [3],
    "AnyJob.n_datapoints": [10**8],
    "AnyJob.p_sample": [0.25],
}

experiments_list = create_experiments_helper(
    experiment_name="eval k = {3} small",
    project_name="pmtest/subgoal-chess",
    base_config=base_config,
    params_grid=params_grid,
    script="python3 -m runner --mrunner",
    exclude=["data", ".pytest_cache", "alpacka.egg-info", "out", ".git"],
    python_path="",
    tags=["subgoal-quality"],
    with_neptune=True,
    env={},
)
