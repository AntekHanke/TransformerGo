from mrunner.helpers.specification_helper import create_experiments_helper

base_config = {
    "run.job_class": "@jobs.LeelaDatasetGenerator",
    "LeelaDatasetGenerator.mcts_gen_class": "@data.SubgoalMCGamesDataGenerator",

    'SubgoalMCGamesDataGenerator.k': 3,
    'SubgoalMCGamesDataGenerator.n_subgoals': 4,
    'SubgoalMCGamesDataGenerator.total_datapoints': 10**8,
    'SubgoalMCGamesDataGenerator.log_samples_limit': 50,
    'SubgoalMCGamesDataGenerator.input_data_dir': "/trees",
    'SubgoalMCGamesDataGenerator.save_data_path': "/save_data/full_dataset",
    "use_neptune": True,
}

params_grid = {
    'SubgoalMCGamesDataGenerator.k': [1,2,3,4,5,6]
}

experiments_list = create_experiments_helper(
    experiment_name="Subgoals form Leela",
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
