from mrunner.helpers.specification_helper import create_experiments_helper

base_config = {
    # job's parameters
    "run.job_class": "@jobs.GoTokenizedPolicyGeneratorAlwaysBlack",
    "GoTokenizedPolicyGeneratorAlwaysBlack.GoTokenizedDataGenerator": "@data.GoSimpleGamesDataGeneratorTokenizedAlwaysBlack",
    # class's parametres (ChessSubgoalGamesDataGenerator)
    "GoSimpleGamesDataGeneratorTokenizedAlwaysBlack.sgf_files": "/plgantekhanke/raw_data/sgfs/test.txt",
    "GoSimpleGamesDataGeneratorTokenizedAlwaysBlack.save_data_every_n_games": 29990,
    "GoSimpleGamesDataGeneratorTokenizedAlwaysBlack.p_sample": 1.0,
    "GoSimpleGamesDataGeneratorTokenizedAlwaysBlack.max_games": 19950,
    "GoSimpleGamesDataGeneratorTokenizedAlwaysBlack.train_eval_split": 0.0,
    "GoSimpleGamesDataGeneratorTokenizedAlwaysBlack.save_path_to_eval_set": "/plgantekhanke/tokenized_data/test/",
    "GoSimpleGamesDataGeneratorTokenizedAlwaysBlack.save_path_to_train_set": "/plgantekhanke/tokenized_data/test/",
    # class's parameters (ELOFilter)

    "use_neptune": True,
}

params_grid = {
    "idx": [0],
    #"GoSimpleGamesDataGeneratorTokenizedAlwaysBlack.max_games": [9950]
}

experiments_list = create_experiments_helper(
    experiment_name=f"generating-tokenized-go-dataset-black",
    project_name="pmtest/subgoal-chess",
    base_config=base_config,
    params_grid=params_grid,
    script="python3 -m runner --mrunner",
    exclude=['sgf', 'val', 'exclude'], #niepotrzebne pliki do clustra
    python_path="",
    tags=["dataset", "generate", "tokenized", "go", "test"],
    with_neptune=True,
    env={},
)
