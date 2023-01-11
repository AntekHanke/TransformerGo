from mrunner.helpers.specification_helper import create_experiments_helper

base_config = {
    # job's parameters
    "run.job_class": "@jobs.CreatePGNDataset",
    "CreatePGNDataset.chess_database_cls": "@data.ChessSubgoalGamesDataGenerator",

    # class's parametres (ChessSubgoalGamesDataGenerator)
    "ChessSubgoalGamesDataGenerator.pgn_file": "/pgn_large/lichess_db_standard_rated_2022-10.pgn",
    "ChessSubgoalGamesDataGenerator.chess_filter": "@filters.ResultFilter",
    "ChessSubgoalGamesDataGenerator.p_sample": 1.0,
    "ChessSubgoalGamesDataGenerator.n_data": 10**10,
    "ChessSubgoalGamesDataGenerator.train_eval_split": 0.95,
    "ChessSubgoalGamesDataGenerator.save_data_path": "/subgoals_dataset_lichess/",
    "ChessSubgoalGamesDataGenerator.save_data_every": 100000,
    "ChessSubgoalGamesDataGenerator.k": 1,

    "use_neptune": True,
}

params_grid = {
    "idx": [0],
    "ChessSubgoalGamesDataGenerator.k": [1],

}

experiments_list = create_experiments_helper(
    experiment_name=f"generating-subgoals-from-lichess-dataset ",
    project_name="pmtest/subgoal-chess",
    base_config=base_config,
    params_grid=params_grid,
    script="python3 -m runner --mrunner",
    exclude=["data", ".pytest_cache", "out", ".git"],
    python_path="",
    tags=["dataset", "generate", "subgolas", "lichess"],
    with_neptune=True,
    env={},
)
