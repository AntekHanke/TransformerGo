from mrunner.helpers.specification_helper import create_experiments_helper

base_config = {
    # job's parameters
    "run.job_class": "@jobs.CreatePGNDataset",
    "CreatePGNDataset.chess_database_cls": "@data.ChessSubgoalGamesDataGenerator",
    # class's parametres (ChessSubgoalGamesDataGenerator)
    "ChessSubgoalGamesDataGenerator.pgn_file": "/database_chess_lichess",
    "ChessSubgoalGamesDataGenerator.p_sample": 1.0,
    "ChessSubgoalGamesDataGenerator.max_games": 10**9,
    "ChessSubgoalGamesDataGenerator.train_eval_split": 0.99,
    "ChessSubgoalGamesDataGenerator.do_sample_finish": False,
    "ChessSubgoalGamesDataGenerator.save_path_to_train_set": "/subgoals_dataset_lichess",
    "ChessSubgoalGamesDataGenerator.save_path_to_eval_set": "/subgoals_dataset_lichess",
    "ChessSubgoalGamesDataGenerator.save_data_every_n_games": 1000,
    "ChessSubgoalGamesDataGenerator.log_stats_after_n_games": 1,
    "ChessSubgoalGamesDataGenerator.only_eval": False,
    "ChessSubgoalGamesDataGenerator.k": 3,
    "ChessSubgoalGamesDataGenerator.number_of_datapoint_from_one_game": 10,
    "ChessSubgoalGamesDataGenerator.chess_filter": "@filters.ELOFilter",
    # class's parameters (ELOFilter)
    "ELOFilter.elo_threshold": 2200,
    "use_neptune": True,
}

params_grid = {
    "idx": [0],
    "ChessSubgoalGamesDataGenerator.k": [3],
    "ChessSubgoalGamesDataGenerator.pgn_file": [
        f"/database_chess_lichess/{file_name}"
        for file_name in [
            "lichess_elite_2013-09.pgn",
            "lichess_elite_2013-11.pgn",
            "lichess_elite_2014-01.pgn",
            "lichess_elite_2014-02.pgn",
            "lichess_elite_2014-03.pgn",
            "lichess_elite_2014-04.pgn",
            "lichess_elite_2014-05.pgn",
            "lichess_elite_2014-06.pgn",
            "lichess_elite_2014-07.pgn",
            "lichess_elite_2014-08.pgn",
            "lichess_elite_2014-09.pgn",
            "lichess_elite_2014-10.pgn",
            "lichess_elite_2014-11.pgn",
            "lichess_elite_2014-12.pgn",
            "lichess_elite_2015-01.pgn",
            "lichess_elite_2015-02.pgn",
            "lichess_elite_2015-03.pgn",
            "lichess_elite_2015-04.pgn",
            "lichess_elite_2015-05.pgn",
            "lichess_elite_2015-06.pgn",
            "lichess_elite_2015-07.pgn",
            "lichess_elite_2015-08.pgn",
            "lichess_elite_2015-09.pgn",
            "lichess_elite_2015-10.pgn",
            "lichess_elite_2015-11.pgn",
            "lichess_elite_2015-12.pgn",
            "lichess_elite_2016-01.pgn",
            "lichess_elite_2016-02.pgn",
            "lichess_elite_2016-03.pgn",
            "lichess_elite_2016-04.pgn",
            "lichess_elite_2016-05.pgn",
            "lichess_elite_2016-06.pgn",
            "lichess_elite_2016-07.pgn",
            "lichess_elite_2016-08.pgn",
            "lichess_elite_2016-09.pgn",
            "lichess_elite_2016-10.pgn",
            "lichess_elite_2016-11.pgn",
            "lichess_elite_2016-12.pgn",
            "lichess_elite_2017-01.pgn",
            "lichess_elite_2017-02.pgn",
            "lichess_elite_2017-03.pgn",
            "lichess_elite_2017-04.pgn",
            "lichess_elite_2017-05.pgn",
            "lichess_elite_2017-06.pgn",
            "lichess_elite_2017-07.pgn",
            "lichess_elite_2017-08.pgn",
            "lichess_elite_2017-09.pgn",
            "lichess_elite_2017-10.pgn",
            "lichess_elite_2017-11.pgn",
            "lichess_elite_2017-12.pgn",
            "lichess_elite_2018-01.pgn",
            "lichess_elite_2018-02.pgn",
            "lichess_elite_2018-03.pgn",
            "lichess_elite_2018-04.pgn",
            "lichess_elite_2018-05.pgn",
            "lichess_elite_2018-06.pgn",
            "lichess_elite_2018-07.pgn",
            "lichess_elite_2018-08.pgn",
            "lichess_elite_2018-09.pgn",
            "lichess_elite_2018-10.pgn",
            "lichess_elite_2018-11.pgn",
            "lichess_elite_2018-12.pgn",
            "lichess_elite_2019-01.pgn",
            "lichess_elite_2019-02.pgn",
            "lichess_elite_2019-03.pgn",
            "lichess_elite_2019-04.pgn",
            "lichess_elite_2019-05.pgn",
            "lichess_elite_2019-06.pgn",
            "lichess_elite_2019-07.pgn",
            "lichess_elite_2019-08.pgn",
            "lichess_elite_2019-09.pgn",
            "lichess_elite_2019-10.pgn",
            "lichess_elite_2019-11.pgn",
            "lichess_elite_2019-12.pgn",
            "lichess_elite_2020-01.pgn",
            "lichess_elite_2020-02.pgn",
            "lichess_elite_2020-03.pgn",
            "lichess_elite_2020-04.pgn",
            "lichess_elite_2020-05.pgn",
            "lichess_elite_2020-06.pgn",
            "lichess_elite_2020-07.pgn",
            "lichess_elite_2020-08.pgn",
            "lichess_elite_2020-09.pgn",
            "lichess_elite_2020-10.pgn",
            "lichess_elite_2020-11.pgn",
            "lichess_elite_2021-01.pgn",
            "lichess_elite_2021-02.pgn",
            "lichess_elite_2021-03.pgn",
            "lichess_elite_2021-04.pgn",
            "lichess_elite_2021-05.pgn",
            "lichess_elite_2021-06.pgn",
            "lichess_elite_2021-07.pgn",
            "lichess_elite_2021-08.pgn",
            "lichess_elite_2021-09.pgn",
            "lichess_elite_2021-10.pgn",
            "lichess_elite_2021-11.pgn",
            "lichess_elite_2021-12.pgn",
            "lichess_elite_2022-02.pgn",
            "lichess_elite_2022-03.pgn",
            "lichess_elite_2022-04.pgn",
            "lichess_elite_2022-05.pgn",
            "lichess_elite_2022-06.pgn",
            "lichess_elite_2022-07.pgn",
            "lichess_elite_2022-08.pgn",
            "lichess_elite_2022-09.pgn",
            "lichess_elite_2022-10.pgn",
            "lichess_elite_2022-11.pgn",
        ]
    ],
}

experiments_list = create_experiments_helper(
    experiment_name=f"generating-subgoals-from-lichess-dataset",
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
