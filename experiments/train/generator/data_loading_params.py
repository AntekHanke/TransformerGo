data_loading_header = {
    "from_scratch": {
        "TrainModelFromScratch.train_data_provider": "@data.PandasIterableSubgoalDataProvider",
        "TrainModelFromScratch.eval_data_provider": "@data.PandasStaticSubgoalDataProvider",
    },
    "resume": {
        "ResumeTraining.train_data_provider": "@data.PandasIterableSubgoalDataProvider",
        "ResumeTraining.eval_data_provider": "@data.PandasStaticSubgoalDataProvider",
    },
}

ultra_small_data = {
    "from_scratch": {
        "TrainModelFromScratch.eval_n_batches": 4,
        "TrainModelFromScratch.path_to_training_data": "/ultra_small_data/train_small.pkl",
        "TrainModelFromScratch.path_to_eval_data": "/ultra_small_data/eval_small.pkl",
        "TrainModelFromScratch.files_batch_size": 5,
    },
    "resume": {
        "ResumeTraining.eval_n_batches": 4,
        "ResumeTraining.path_to_training_data": "/ultra_small_data/train_small.pkl",
        "ResumeTraining.path_to_eval_data": "/ultra_small_data/eval_small.pkl",
        "ResumeTraining.files_batch_size": 5,
    },
}

standard_data_k3 = {
    "from_scratch": {
        "TrainModel.eval_n_batches": 50,
        "TrainModel.path_to_training_data": "/subgoals_dataset_lichess/subgoals_k=3/datapoints_train",
        "TrainModel.path_to_eval_data": "/subgoals_dataset_lichess/subgoals_k=3/datapoints_eval",
        "TrainModel.files_batch_size": 1000,
    },
    "resume": {
        "ResumeTraining.eval_n_batches": 50,
        "ResumeTraining.path_to_training_data": "/subgoals_dataset_lichess/subgoals_k=3/datapoints_train",
        "ResumeTraining.path_to_eval_data": "/subgoals_dataset_lichess/subgoals_k=3/datapoints_eval",
        "ResumeTraining.files_batch_size": 1000,
    },
}
