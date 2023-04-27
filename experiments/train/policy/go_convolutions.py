from mrunner.helpers.specification_helper import create_experiments_helper


base_config = {
    "run.job_class": "@jobs.GoTrainConvolution",
    "GoTrainConvolution.go_database_cls": "@data.PandasIterablePolicyDataProviderGo",
    "GoTrainConvolution.model_config_cls": "@AlphaZero.AlphaZeroPolicyConfig",
    "GoTrainConvolution.training_args_cls": "@transformers.TrainingArguments",
    "PandasIterablePolicyDataProviderGo.data_path": "./data_processing/jgdb/tokenizeddata/trainval_train_part_0.pkl",
    "PandasBertForSequenceDataProvider.eval_datapoints": 20,
    "GlobalParamsHandler.out_dir": "./jobs/local_jobs_mgrot/policytrain",
    # "GlobalParamsHandler.data_location": "/leela_bert_data/prom_full_dataset_k=3.pkl",
    # # "GlobalParamsHandler.data_type": "cllp",
    # "GlobalParamsHandler.path_format": ["learning_rate"],
    "alphaZeroPolicyConfig.num_residual_blocks" : 19,
    "AlphaZeroPolicyConfig.num_in_channels" : 5,
    "AlphaZeroPolicyConfig.num_out_channels" : 256,
    "AlphaZeroPolicyConfig.kernel_size" : 3,
    "AlphaZeroPolicyConfig.stride" : 1,
    "AlphaZeroPolicyConfig.board_size" : (19, 19),   
    "TrainingArguments.num_train_epochs": 1,
    "TrainingArguments.per_device_train_batch_size": 20,
    "TrainingArguments.per_device_eval_batch_size": 20,
    "TrainingArguments.warmup_steps": 100,
    "TrainingArguments.weight_decay": 0.01,
    "TrainingArguments.logging_steps": 10,
    "TrainingArguments.evaluation_strategy": "steps",
    "TrainingArguments.eval_steps": 5,
    "TrainingArguments.learning_rate": 0.0002,
    "use_neptune": True,
}

params_grid = {
    "idx": [0],
    # "GlobalParamsHandler.k": [3]
}

experiments_list = create_experiments_helper(
    experiment_name=f"conv-local-test",
    project_name="mgrotkowski/debug-project",
    base_config=base_config,
    params_grid=params_grid,
    script="python3 -m runner --mrunner",
    exclude=["data", ".pytest_cache", "out", ".git"],
    python_path="",
    tags=["Convolutions", "train", "ultra-small"],
    with_neptune=True,
    env={},
)
