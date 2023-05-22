from mrunner.helpers.specification_helper import create_experiments_helper

base_config = {
    # job's parameters
    "run.job_class": "@jobs.GoTokenizedSubgoalGenerator",
    "GoTokenizedPolicyGeneratorAlwaysBlack.GoTokenizedDataGenerator": "@data.GoSubgoalGamesDataGenerator",
    # class's parametres (ChessSubgoalGamesDataGenerator)
    "GoSubgoalGamesDataGenerator.sgf_files": "/plgantekhanke/raw_data/aio_games/2005-11.aio",
    "GoSubgoalGamesDataGenerator.save_data_every_n_games": 29990,
    "GoSubgoalGamesDataGenerator.p_sample": 1.0,
    "GoSubgoalGamesDataGenerator.max_games": 19950,
    "GoSubgoalGamesDataGenerator.train_eval_split": 0.0,
    "GoSubgoalGamesDataGenerator.save_path_to_eval_set": "/plgantekhanke/tokenized_data/subgoal/testing/",
    "GoSubgoalGamesDataGenerator.save_path_to_train_set": "/plgantekhanke/tokenized_data/subgoal/testing/",
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
    tags=["dataset", "generate", "tokenized", "go", "test", "subgoals"],
    with_neptune=True,
    env={},
)
