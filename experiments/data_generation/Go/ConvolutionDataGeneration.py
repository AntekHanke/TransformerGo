from mrunner.helpers.specification_helper import create_experiments_helper

base_config = {
    # job's parameters
    "run.job_class": "@jobs.GoConvolutionDataGeneration",
    "GoConvolutionDataGeneration.GoConvolutionDataGenerator": "@data.SimpleGamesDataGeneratorWithHistory",
    "SimpleGamesDataGeneratorWithHistory.save_data_every_n_games": 999,
    "SimpleGamesDataGeneratorWithHistory.p_sample": 1.0,
    "SimpleGamesDataGeneratorWithHistory.max_games":  999,
    "SimpleGamesDataGeneratorWithHistory.train_eval_split": 0,
    "SimpleGamesDataGeneratorWithHistory.save_path_to_eval_set": "/godata/val_with_history/",
    "SimpleGamesDataGeneratorWithHistory.save_path_to_train_set": "/godata/val_with_history/",
    # class's parameters (ELOFilter)

    "use_neptune": True,
}
PATH_STR = "/godata/validx/validx_"
params_grid = {
    "idx": [0],
    "SimpleGamesDataGenerator.sgf_files" : [PATH_STR + str(0) + str(i) + ".txt"
                                            if i < 10 else PATH_STR + str(i) + ".txt" for i in range(1,11)] 
    #"SimpleGamesDataGenerator.sgf_files" : [PATH_STR + "01" ".txt"] 
}

experiments_list = create_experiments_helper(
    experiment_name=f"train-go-dataset-with-history-val",
    project_name="pmtest/subgoal-chess",
    base_config=base_config,
    params_grid=params_grid,
    script="python3 -m runner --mrunner",
    exclude=['sgf', 'val', 'exclude', 'lib', 'lib64', 'model_checkpoints'] , #niepotrzebne pliki do clustra
    python_path="",
    tags=["dataset", "generate", "convolution", "go"],
    with_neptune=True,
    env={},
)
