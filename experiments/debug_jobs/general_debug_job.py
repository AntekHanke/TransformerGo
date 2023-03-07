from datetime import date

from mrunner.helpers.specification_helper import create_experiments_helper

base_config = {
    "run.job_class": "@jobs.DebugJob",
    "use_neptune": True,
}

params_grid = {}

experiments_list = create_experiments_helper(
    experiment_name="testlczero",
    project_name="pmtest/subgoal-chess",
    base_config=base_config,
    params_grid=params_grid,
    script="python3 -m runner --mrunner",
    exclude=["data", ".pytest_cache", "out", ".git"],
    python_path="",
    tags=["leela", "train", "medium", "subgoals", "batch_1100"],
    with_neptune=True,
    env={},
)
