subgoal_data_loading_header = {
    "from_scratch": {
        "TrainModelFromScratch.train_data_provider": "@data.PandasIterableSubgoalDataProvider",
        "TrainModelFromScratch.eval_data_provider": "@data.PandasStaticSubgoalDataProvider",
    },
    "from_scratch_with_all_subgoals": {
        "TrainModelFromScratch.train_data_provider": "@data.PandasIterableSubgoalAllDistancesDataProvider",
        "TrainModelFromScratch.eval_data_provider": "@data.PandasStaticSubgoalAllDistancesDataProvider",
    },
    "resume": {
        "ResumeTraining.train_data_provider": "@data.PandasIterableSubgoalDataProvider",
        "ResumeTraining.eval_data_provider": "@data.PandasStaticSubgoalDataProvider",
    },
    "resume_with_all_subgoals": {
        "ResumeTraining.train_data_provider": "@data.PandasIterableSubgoalAllDistancesDataProvider",
        "ResumeTraining.eval_data_provider": "@data.PandasStaticSubgoalAllDistancesDataProvider",
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

policy_data_loading_header = {
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

policy_data_loading_headerGo = {
    "from_scratch": {
        "TrainModelFromScratch.train_data_provider": "@data.PandasIterablePolicyDataProviderGo",
        "TrainModelFromScratch.eval_data_provider": "@data.PandasStaticPolicyDataProviderGo",
    },
}


ultra_small_data = {
    "from_scratch": {
        "TrainModelFromScratch.eval_n_batches": 4,
        "TrainModelFromScratch.path_to_training_data": "/ultra_small_data/train_small.pkl",
        "TrainModelFromScratch.path_to_eval_data": "/ultra_small_data/eval_small.pkl",
        "TrainModelFromScratch.files_batch_size": 5,
    },
    "from_scratch_with_all_subgoals": {
        "TrainModelFromScratch.eval_n_batches": 4,
        "TrainModelFromScratch.path_to_training_data": "/ultra_small_data/train",
        "TrainModelFromScratch.path_to_eval_data": "/ultra_small_data/eval",
        "TrainModelFromScratch.files_batch_size": 5,
        "TrainModelFromScratch.range_of_k": [1, 2, 3],
    },
    "resume": {
        "ResumeTraining.eval_n_batches": 4,
        "ResumeTraining.path_to_training_data": "/ultra_small_data/train_small.pkl",
        "ResumeTraining.path_to_eval_data": "/ultra_small_data/eval_small.pkl",
        "ResumeTraining.files_batch_size": 5,
    },
    "resume_with_all_subgoals": {
        "ResumeTraining.eval_n_batches": 4,
        "ResumeTraining.path_to_training_data": "/ultra_small_data/train",
        "ResumeTraining.path_to_eval_data": "/ultra_small_data/eval",
        "ResumeTraining.files_batch_size": 5,
        "ResumeTraining.range_of_k": [1, 2, 3, 4, 5, 6],
    },
}

go_ultra_small_data = {
    "from_scratch": {
        "TrainModelFromScratch.eval_n_batches": 4,
        "TrainModelFromScratch.path_to_training_data": "/plgantekhanke/sgfs/tokenizeddata/trainval_train_part_0.pkl",
        "TrainModelFromScratch.path_to_eval_data": "/plgantekhanke/sgfs/tokenizeddata/evalval_eval_part_0.pkl",
        "TrainModelFromScratch.files_batch_size": 5,
    },

}
go_medium_data = {
    "from_scratch": {
        "TrainModelFromScratch.eval_n_batches": 10,
        "TrainModelFromScratch.path_to_training_data": "/plgantekhanke/sgfs/tokenizeddata/trainval_train_part_0.pkl",
        "TrainModelFromScratch.path_to_eval_data": "/plgantekhanke/sgfs/tokenizeddata/evalval_eval_part_0.pkl",
        "TrainModelFromScratch.files_batch_size": 10,
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

STANDARD_WHOLE_DATA_TRAIN_BOARD = "/board/subgoals_all_k/train"
STANDARD_WHOLE_DATA_EVAL_BOARD = "/board/subgoals_all_k/eval"
standard_whole_data_board_tokenizer = {
    "from_scratch": {
        "TrainModelFromScratch.eval_n_batches": 50,
        "TrainModelFromScratch.path_to_training_data": STANDARD_WHOLE_DATA_TRAIN_BOARD,
        "TrainModelFromScratch.path_to_eval_data": STANDARD_WHOLE_DATA_EVAL_BOARD,
        "TrainModelFromScratch.files_batch_size": 1000,
    },
    "resume": {
        "ResumeTraining.eval_n_batches": 50,
        "ResumeTraining.path_to_training_data": STANDARD_WHOLE_DATA_TRAIN_BOARD,
        "ResumeTraining.path_to_eval_data": STANDARD_WHOLE_DATA_EVAL_BOARD,
        "ResumeTraining.files_batch_size": 1000,
    },
    "from_scratch_with_all_subgoals": {
        "TrainModelFromScratch.eval_n_batches": 50,
        "TrainModelFromScratch.path_to_training_data": STANDARD_WHOLE_DATA_TRAIN_BOARD,
        "TrainModelFromScratch.path_to_eval_data": STANDARD_WHOLE_DATA_EVAL_BOARD,
        "TrainModelFromScratch.files_batch_size": 1000,
        "TrainModelFromScratch.range_of_k": [1, 2, 3, 4, 5, 6],
    },
    "resume_with_all_subgoals": {
        "ResumeTraining.eval_n_batches": 50,
        "ResumeTraining.path_to_training_data": STANDARD_WHOLE_DATA_TRAIN_BOARD,
        "ResumeTraining.path_to_eval_data": STANDARD_WHOLE_DATA_EVAL_BOARD,
        "ResumeTraining.files_batch_size": 1000,
        "ResumeTraining.range_of_k": [1, 2, 3, 4, 5, 6],
    },
}