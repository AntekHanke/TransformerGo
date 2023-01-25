from mrunner.helpers.specification_helper import create_experiments_helper

base_config = {
    # job's parameters
    "run.job_class": "@jobs.CreatePGNDataset",
    "CreatePGNDataset.chess_database_cls": "@data.ChessSubgoalGamesDataGenerator",

    # class's parametres (ChessSubgoalGamesDataGenerator)
    "ChessSubgoalGamesDataGenerator.pgn_file": "/home/tomasz/Research/subgoal_search_chess/assets/cas_small.pgn",
    "ChessSubgoalGamesDataGenerator.chess_filter": "@filters.ELOFilter",
    "ChessSubgoalGamesDataGenerator.p_sample": 1.0,
    "ChessSubgoalGamesDataGenerator.max_games": 9,
    "ChessSubgoalGamesDataGenerator.train_eval_split": 1.0,
    "ChessSubgoalGamesDataGenerator.do_sample_finish": False,
    "ChessSubgoalGamesDataGenerator.log_stats_after_n_games": 1,
    "ChessSubgoalGamesDataGenerator.save_data_path": "/home/tomasz/Research/subgoal_chess_data/temp/main_data/",
    "ChessSubgoalGamesDataGenerator.save_filtered_data": "/home/tomasz/Research/subgoal_chess_data/temp/main_data/filtered_data",
    "ChessSubgoalGamesDataGenerator.save_data_every_n_games": 4,

    "ChessSubgoalGamesDataGenerator.k": 3,
    "ChessSubgoalGamesDataGenerator.number_of_datapoint_from_one_game": 5,


    # class's parameters (ELOFilter)
    "ELOFilter.elo_threshold": 150,

    "use_neptune": False,
}

params_grid = {
    "idx": [0],
    # "ChessSubgoalGamesDataGenerator.k": [2, 3],

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