from mrunner.helpers.specification_helper import create_experiments_helper

base_config = {
    # job's parameters
    "run.job_class": "@jobs.GoTokenizedPolicyGeneratorAlwaysBlack",
    "GoTokenizedPolicyGeneratorAlwaysBlack.GoTokenizedDataGenerator": "@data.GoSimpleGamesDataGeneratorTokenizedAlwaysBlack",
    # class's parametres (ChessSubgoalGamesDataGenerator)
    "GoSimpleGamesDataGeneratorTokenizedAlwaysBlack.sgf_files": "/net/scratch/people/plgantekhanke/sgfs/val.txt",
    "GoSimpleGamesDataGeneratorTokenizedAlwaysBlack.save_data_every_n_games": 990,
    "GoSimpleGamesDataGeneratorTokenizedAlwaysBlack.p_sample": 1.0,
    "GoSimpleGamesDataGeneratorTokenizedAlwaysBlack.max_games": 9950,
    "GoSimpleGamesDataGeneratorTokenizedAlwaysBlack.train_eval_split": 0.95,
    "GoSimpleGamesDataGeneratorTokenizedAlwaysBlack.save_path_to_eval_set": "/net/scratch/people/plgantekhanke/sgfs/tokenizeddata/eval",
    "GoSimpleGamesDataGeneratorTokenizedAlwaysBlack.save_path_to_train_set": "/net/scratch/people/plgantekhanke/sgfs/tokenizeddata/train",
    # class's parameters (ELOFilter)

    "use_neptune": True,
}

params_grid = {
    "idx": [0],
    "GoSimpleGamesDataGeneratorTokenizedAlwaysBlack.max_games": [9950]
}

experiments_list = create_experiments_helper(
    experiment_name=f"generating-tokenized-go-dataset-black",
    project_name="pmtest/subgoal-chess",
    base_config=base_config,
    params_grid=params_grid,
    script="python3 -m runner --mrunner",
    exclude=['sgf','data_processing', 'val'], #niepotrzebne pliki do clustra
    python_path="",
    tags=["dataset", "generate", "tokenized", "go"],
    with_neptune=True,
    env={},
)
