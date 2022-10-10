from mrunner.helpers.specification_helper import create_experiments_helper

base_config = {
}

params_grid = {
    "idx": [0]
}

experiments_list = create_experiments_helper(
    experiment_name="Subgoals form Leela",
    project_name="pmtest/subgoal-chess",
    base_config=base_config,
    params_grid=params_grid,
    script="python3 read_tmp.py",
    exclude=["data", ".pytest_cache", "alpacka.egg-info", "out", ".git"],
    python_path="",
    tags=[globals()["script"][:-3], "subgoals_from_leela"],
    with_neptune=True,
    env={},
)
