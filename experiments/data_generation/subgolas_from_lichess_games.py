from mrunner.helpers.specification_helper import create_experiments_helper

base_config = {
    # job's parameters
    "run.job_class": "@jobs.CreatePGNDataset",
    "CreatePGNDataset.chess_database_cls": "@data.ChessSubgoalGamesDataGenerator",
    # class's parametres (ChessSubgoalGamesDataGenerator)
    "ChessSubgoalGamesDataGenerator.pgn_file": "/pgn_large/chess_engines_gameplayes.pgn",
    "ChessSubgoalGamesDataGenerator.p_sample": 1.0,
    "ChessSubgoalGamesDataGenerator.n_data": 10**8,
    "ChessSubgoalGamesDataGenerator.train_eval_split": 0.95,
    "ChessSubgoalGamesDataGenerator.save_data_path": "/subgoals_dataset_lichess/",
    "ChessSubgoalGamesDataGenerator.save_data_every": 10000,
    "ChessSubgoalGamesDataGenerator.log_stats_after_n": 100,
    "ChessSubgoalGamesDataGenerator.k": 1,
    "ChessSubgoalGamesDataGenerator.number_of_datapoint_from_one_game": 2,
    "ChessSubgoalGamesDataGenerator.chess_filter": "@filters.ELOFilter",
    # class's parameters (ELOFilter)
    "ELOFilter.elo_threshold": 1500,
    "use_neptune": True,
}

params_grid = {
    "idx": [0],
    "ChessSubgoalGamesDataGenerator.k": [3],
    "ChessSubgoalGamesDataGenerator.pgn_file": [
        f"/database_chess_lichess/splited_data/{name_of_file}"
        for name_of_file in [
            "chess_engines_gameplayes_aa",
            "chess_engines_gameplayes_ab",
            "chess_engines_gameplayes_ac",
            "chess_engines_gameplayes_ad",
            "chess_engines_gameplayes_ae",
            "chess_engines_gameplayes_af",
            "chess_engines_gameplayes_ag",
            "chess_engines_gameplayes_ah",
            "chess_engines_gameplayes_ai",
            "chess_engines_gameplayes_aj",
            "chess_engines_gameplayes_ak",
            "chess_engines_gameplayes_al",
            "chess_engines_gameplayes_am",
            "chess_engines_gameplayes_an",
            "chess_engines_gameplayes_ao",
            "chess_engines_gameplayes_ap",
            "chess_engines_gameplayes_aq",
            "chess_engines_gameplayes_ar",
            "chess_engines_gameplayes_as",
            "chess_engines_gameplayes_at",
            "chess_engines_gameplayes_au",
            "chess_engines_gameplayes_av",
            "chess_engines_gameplayes_aw",
            "chess_engines_gameplayes_ax",
            "chess_engines_gameplayes_ay",
            "chess_engines_gameplayes_az",
            "chess_engines_gameplayes_ba",
            "chess_engines_gameplayes_bb",
            "chess_engines_gameplayes_bc",
            "chess_engines_gameplayes_bd",
            "chess_engines_gameplayes_be",
            "chess_engines_gameplayes_bf",
            "chess_engines_gameplayes_bg",
            "chess_engines_gameplayes_bh",
            "chess_engines_gameplayes_bi",
            "chess_engines_gameplayes_bj",
            "chess_engines_gameplayes_bk",
            "chess_engines_gameplayes_bl",
            "chess_engines_gameplayes_bm",
        ]
    ],
}

experiments_list = create_experiments_helper(
    experiment_name=f"generating-subgoals-from-lichess-dataset-test ",
    project_name="pmtest/subgoal-chess",
    base_config=base_config,
    params_grid=params_grid,
    script="python3 -m runner --mrunner",
    exclude=["data", ".pytest_cache", "out", ".git"],
    python_path="",
    tags=["dataset", "generate", "subgolas", "lichess", "test"],
    with_neptune=True,
    env={},
)
