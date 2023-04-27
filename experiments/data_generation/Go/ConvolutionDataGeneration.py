from mrunner.helpers.specification_helper import create_experiments_helper

base_config = {
    # job's parameters
    "run.job_class": "@jobs.GoConvolutionDataGeneration",
    "GoConvolutionDataGeneration.GoConvolutionDataGenerator": "@data.SimpleGamesDataGenerator",
    #"SimpleGamesDataGenerator.sgf_files": "/home/users/mgrotkowski/grant_619/scratch/AH/raw_data/val.txt",
    "SimpleGamesDataGenerator.save_data_every_n_games": 999,
    "SimpleGamesDataGenerator.p_sample": 1.0,
    "SimpleGamesDataGenerator.max_games": 999,
    "SimpleGamesDataGenerator.train_eval_split": 0.95,
    "SimpleGamesDataGenerator.save_path_to_eval_set": "/godata/val/",
    "SimpleGamesDataGenerator.save_path_to_train_set": "/godata/val/",
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
    experiment_name=f"generating-go-dataset",
    project_name="mgrotkowski/debug-project",
    base_config=base_config,
    params_grid=params_grid,
    script="python3 -m runner --mrunner",
    exclude=['sgf', 'val', 'exclude', 'lib', 'lib64'] , #niepotrzebne pliki do clustra
    python_path="",
    tags=["dataset", "generate", "convolution", "go"],
    with_neptune=True,
    env={},
)
