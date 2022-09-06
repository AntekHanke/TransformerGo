from mrunner.helpers.specification_helper import create_experiments_helper

base_config = {"run.job_class": "@jobs.AnyJob",
               "use_neptune": True}

params_grid = {
    "idx": [0],
    "AnyJob.learning_rate": [3e-4],
    "AnyJob.k": [2,3,4],
    "AnyJob.n_datapoints": [5*10**7],
    "AnyJob.p_sample": [0.25]
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
