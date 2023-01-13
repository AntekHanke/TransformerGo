from mrunner.helpers.specification_helper import create_experiments_helper

base_config = {
    "run.job_class": "@jobs.CreatePGNDataset",
    "CreatePGNDataset.chess_database_cls": "@data.ChessCLLPGamesDataGenerator",
    "ChessCLLPGamesDataGenerator.max_k": 6,
    "ChessCLLPGamesDataGenerator.pgn_file": "/chess_data/pgn/database.pgn",
    "ChessCLLPGamesDataGenerator.chess_filter": "@filters.NoFilter",
    "ChessCLLPGamesDataGenerator.p_sample": 0.1,
    "ChessCLLPGamesDataGenerator.n_data": 10**4,
    "ChessCLLPGamesDataGenerator.log_samples_limit": 100,
    "ChessCLLPGamesDataGenerator.p_log_sample": 0.01,
    "ChessCLLPGamesDataGenerator.save_data_path": "/chess_data/pgn/train_data/cllp/",
    "ChessCLLPGamesDataGenerator.save_data_every": 100,
    "use_neptune": True,
}

params_grid = {"idx": [0]}

experiments_list = create_experiments_helper(
    experiment_name="cllp pgn",
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
