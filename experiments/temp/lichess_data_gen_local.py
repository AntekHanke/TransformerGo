from mrunner.helpers.specification_helper import create_experiments_helper

base_config = {
    # job's parameters
    "run.job_class": "@jobs.CreatePGNDataset",
    "CreatePGNDataset.chess_database_cls": "@data.ChessSubgoalGamesDataGenerator",

    # class's parametres (ChessSubgoalGamesDataGenerator)
    "ChessSubgoalGamesDataGenerator.pgn_file": "/home/tomasz/Research/subgoal_search_chess/assets/cas_small.pgn",
    "ChessSubgoalGamesDataGenerator.p_sample": 1.0,
    "ChessSubgoalGamesDataGenerator.n_data": 10000,
    "ChessSubgoalGamesDataGenerator.train_eval_split": 0.5,
    "ChessSubgoalGamesDataGenerator.do_sample_finish": True,
    "ChessSubgoalGamesDataGenerator.save_data_path": "/home/tomasz/Research/subgoal_chess_data/temp/",
    "ChessSubgoalGamesDataGenerator.save_data_every": 1,
    "ChessSubgoalGamesDataGenerator.log_stats_after_n": 4,
    "ChessSubgoalGamesDataGenerator.k": 2,
    "ChessSubgoalGamesDataGenerator.number_of_datapoint_from_one_game": 3,
    "ChessSubgoalGamesDataGenerator.chess_filter": "@filters.ELOFilter",

    # class's parameters (ELOFilter)
    "ELOFilter.elo_threshold": 1500,

    "use_neptune": False,
}

params_grid = {
    "idx": [0],
    # "ChessSubgoalGamesDataGenerator.k": [1],

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
    with_neptune=False,
    env={},
)