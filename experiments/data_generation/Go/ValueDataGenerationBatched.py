from mrunner.helpers.specification_helper import create_experiments_helper

base_config = {
    # job's parameters
    "run.job_class": "@jobs.GoTokenizedValueGenerator",
    "GoTokenizedValueGenerator.GoTokenizedDataGenerator": "@data.GoValueTokenized",
    # class's parametres (ChessSubgoalGamesDataGenerator)
    #"GoValueTokenized.sgf_files": "/plgantekhanke/raw_data/sgfs/test.txt",
    "GoValueTokenized.save_data_every_n_games": 29990,
    "GoValueTokenized.p_sample": 1.0,
    "GoValueTokenized.max_games": 15000,
    "GoValueTokenized.train_eval_split": 1.0,
    "GoValueTokenized.save_path_to_eval_set": "/plgantekhanke/tokenized_value_data/train/",
    "GoValueTokenized.save_path_to_train_set": "/plgantekhanke/tokenized_value_data/train/",
    # class's parameters (ELOFilter)

    "use_neptune": True,
}



params_grid = {
    "GoValueTokenized.sgf_files": ["/plgantekhanke/raw_data/sgfs/training_data_01.txt", "/plgantekhanke/raw_data/sgfs/training_data_02.txt", "/plgantekhanke/raw_data/sgfs/training_data_03.txt", "/plgantekhanke/raw_data/sgfs/training_data_04.txt", "/plgantekhanke/raw_data/sgfs/training_data_05.txt", "/plgantekhanke/raw_data/sgfs/training_data_06.txt", "/plgantekhanke/raw_data/sgfs/training_data_07.txt", "/plgantekhanke/raw_data/sgfs/training_data_08.txt", "/plgantekhanke/raw_data/sgfs/training_data_09.txt", "/plgantekhanke/raw_data/sgfs/training_data_10.txt", "/plgantekhanke/raw_data/sgfs/training_data_11.txt", "/plgantekhanke/raw_data/sgfs/training_data_12.txt", "/plgantekhanke/raw_data/sgfs/training_data_13.txt", "/plgantekhanke/raw_data/sgfs/training_data_14.txt", "/plgantekhanke/raw_data/sgfs/training_data_15.txt", "/plgantekhanke/raw_data/sgfs/training_data_16.txt", "/plgantekhanke/raw_data/sgfs/training_data_17.txt", "/plgantekhanke/raw_data/sgfs/training_data_18.txt", "/plgantekhanke/raw_data/sgfs/training_data_19.txt", "/plgantekhanke/raw_data/sgfs/training_data_20.txt", "/plgantekhanke/raw_data/sgfs/training_data_21.txt", "/plgantekhanke/raw_data/sgfs/training_data_22.txt", "/plgantekhanke/raw_data/sgfs/training_data_23.txt", "/plgantekhanke/raw_data/sgfs/training_data_24.txt", "/plgantekhanke/raw_data/sgfs/training_data_25.txt", "/plgantekhanke/raw_data/sgfs/training_data_26.txt", "/plgantekhanke/raw_data/sgfs/training_data_27.txt", "/plgantekhanke/raw_data/sgfs/training_data_28.txt", "/plgantekhanke/raw_data/sgfs/training_data_29.txt", "/plgantekhanke/raw_data/sgfs/training_data_30.txt", "/plgantekhanke/raw_data/sgfs/training_data_31.txt", "/plgantekhanke/raw_data/sgfs/training_data_32.txt", "/plgantekhanke/raw_data/sgfs/training_data_33.txt", "/plgantekhanke/raw_data/sgfs/training_data_34.txt", "/plgantekhanke/raw_data/sgfs/training_data_35.txt", "/plgantekhanke/raw_data/sgfs/training_data_36.txt", "/plgantekhanke/raw_data/sgfs/training_data_37.txt", "/plgantekhanke/raw_data/sgfs/training_data_38.txt", "/plgantekhanke/raw_data/sgfs/training_data_39.txt", "/plgantekhanke/raw_data/sgfs/training_data_40.txt", "/plgantekhanke/raw_data/sgfs/training_data_41.txt", "/plgantekhanke/raw_data/sgfs/training_data_42.txt", "/plgantekhanke/raw_data/sgfs/training_data_43.txt", "/plgantekhanke/raw_data/sgfs/training_data_44.txt", "/plgantekhanke/raw_data/sgfs/training_data_45.txt", "/plgantekhanke/raw_data/sgfs/training_data_46.txt", "/plgantekhanke/raw_data/sgfs/training_data_47.txt", "/plgantekhanke/raw_data/sgfs/training_data_48.txt", "/plgantekhanke/raw_data/sgfs/training_data_49.txt", "/plgantekhanke/raw_data/sgfs/training_data_50.txt"],

    #"GoSimpleGamesDataGeneratorTokenizedAlwaysBlack.sgf_files": ["/plgantekhanke/raw_data/sgfs/training_data_01.txt", "/plgantekhanke/raw_data/sgfs/training_data_02.txt"]
    #"GoSimpleGamesDataGeneratorTokenizedAlwaysBlack.max_games": [9950]
}

experiments_list = create_experiments_helper(
    experiment_name=f"generating-tokenized-go-dataset-black-batched",
    project_name="pmtest/subgoal-chess",
    base_config=base_config,
    params_grid=params_grid,
    script="python3 -m runner --mrunner",
    exclude=['sgf', 'val', 'exclude'], #niepotrzebne pliki do clustra
    python_path="",
    tags=["dataset", "generate", "tokenized", "go", "train", "JGDB", "batched", "value"],
    with_neptune=True,
    env={},
)
