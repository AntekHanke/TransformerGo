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

STANDARD_DATA_K_3_TRAIN = "/data_to/data/k_3"
STANDARD_DATA_K_3_EVAL = "/data_to/data/k_3_eval"
standard_data_k3_board_tokenizer = {
    "from_scratch": {
        "TrainModelFromScratch.eval_n_batches": 50,
        "TrainModelFromScratch.path_to_training_data": STANDARD_DATA_K_3_TRAIN,
        "TrainModelFromScratch.path_to_eval_data": STANDARD_DATA_K_3_EVAL,
        "TrainModelFromScratch.files_batch_size": 2000,
    },
    "resume": {
        "ResumeTraining.eval_n_batches": 50,
        "ResumeTraining.path_to_training_data": STANDARD_DATA_K_3_TRAIN,
        "ResumeTraining.path_to_eval_data": STANDARD_DATA_K_3_EVAL,
        "ResumeTraining.files_batch_size": 2000,
    },
}

STANDARD_DATA_K_3_TRAIN_PIECES = "/data_mg/data/k_3"
STANDARD_DATA_K_3_EVAL_PIECES = "/data_mg/data/k_3_eval"
standard_data_k3_pieces_tokenizer = {
    "from_scratch": {
        "TrainModelFromScratch.eval_n_batches": 50,
        "TrainModelFromScratch.path_to_training_data": STANDARD_DATA_K_3_TRAIN_PIECES,
        "TrainModelFromScratch.path_to_eval_data": STANDARD_DATA_K_3_EVAL_PIECES,
        "TrainModelFromScratch.files_batch_size": 1000,
    },
    "resume": {
        "ResumeTraining.eval_n_batches": 50,
        "ResumeTraining.path_to_training_data": STANDARD_DATA_K_3_TRAIN_PIECES,
        "ResumeTraining.path_to_eval_data": STANDARD_DATA_K_3_EVAL_PIECES,
        "ResumeTraining.files_batch_size": 1000,
    },
}