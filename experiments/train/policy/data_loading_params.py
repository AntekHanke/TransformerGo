data_loading_header = {
    "from_scratch": {
        "TrainModelFromScratch.train_data_provider": "@data.PandasIterablePolicyDataProvider",
        "TrainModelFromScratch.eval_data_provider": "@data.PandasStaticPolicyDataProvider",
    },
    "resume": {
        "ResumeTraining.train_data_provider": "@data.PandasIterablePolicyDataProvider",
        "ResumeTraining.eval_data_provider": "@data.PandasStaticPolicyDataProvider",
    },
    "from_scratch_with_history": {
        "TrainModelFromScratch.train_data_provider": "@data.PandasIterablePolicyWithHistoryDataProvider",
        "TrainModelFromScratch.eval_data_provider": "@data.PandasStaticPolicyWithHistoryDataProvider",
    },
    "resume_with_history": {
        "ResumeTraining.train_data_provider": "@data.PandasIterablePolicyWithHistoryDataProvider",
        "ResumeTraining.eval_data_provider": "@data.PandasStaticPolicyWithHistoryDataProvider",
    },
}

ultra_small_data = {
    "from_scratch": {
        "TrainModelFromScratch.eval_n_batches": 4,
        "TrainModelFromScratch.path_to_training_data": "/ultra_small_data/train_small.pkl",
        "TrainModelFromScratch.path_to_eval_data": "/ultra_small_data/eval_small.pkl",
        "TrainModelFromScratch.files_batch_size": 1,
    },
    "resume": {
        "ResumeTraining.eval_n_batches": 4,
        "ResumeTraining.path_to_training_data": "/ultra_small_data/train_small.pkl",
        "ResumeTraining.path_to_eval_data": "/ultra_small_data/eval_small.pkl",
        "ResumeTraining.files_batch_size": 1,
    },
}

standard_data_k3 = {
    "from_scratch": {
        "TrainModel.eval_n_batches": 1000,
        "TrainModel.path_to_training_data": "/subgoals_dataset_lichess/subgoals_k=3/datapoints_train",
        "TrainModel.path_to_eval_data": "/subgoals_dataset_lichess/subgoals_k=3/datapoints_eval",
        "TrainModel.files_batch_size": 1000,
    },
    "resume": {
        "ResumeTraining.eval_n_batches": 1000,
        "ResumeTraining.path_to_training_data": "/subgoals_dataset_lichess/subgoals_k=3/datapoints_train",
        "ResumeTraining.path_to_eval_data": "/subgoals_dataset_lichess/subgoals_k=3/datapoints_eval",
        "ResumeTraining.files_batch_size": 1000,
    },
}
