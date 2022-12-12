from mrunner.helpers.specification_helper import create_experiments_helper

base_config = {
    "run.job_class": "@jobs.LeelaDatasetGenerator",
    "LeelaDatasetGenerator.mcts_gen_class": "@data.SubgoalMCGamesDataGenerator",
    "SubgoalMCGamesDataGenerator.k": 1,
    "SubgoalMCGamesDataGenerator.n_subgoals": 1,
    "SubgoalMCGamesDataGenerator.total_datapoints": 10 ** 9,
    "SubgoalMCGamesDataGenerator.log_samples_limit": 50,
    "SubgoalMCGamesDataGenerator.input_data_dir": "/leela_data",
    "SubgoalMCGamesDataGenerator.save_data_path": "/save_data/policy_dataset",
    "SubgoalMCGamesDataGenerator.save_data_every": 5000,
    "use_neptune": True,
}

params_grid = {"SubgoalMCGamesDataGenerator.k": [1]}

experiments_list = create_experiments_helper(
    experiment_name="prom large subgoals form Leela",
    project_name="pmtest/subgoal-chess",
    base_config=base_config,
    params_grid=params_grid,
    script="python3 -m runner --mrunner",
    exclude=["data", ".pytest_cache", "alpacka.egg-info", "out", ".git"],
    python_path="",
    tags=[globals()["script"][:-3], "subgoals_from_leela"],
    with_neptune=True,
    env={},
)
