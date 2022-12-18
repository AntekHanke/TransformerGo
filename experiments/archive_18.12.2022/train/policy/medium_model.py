from mrunner.helpers.specification_helper import create_experiments_helper

#Lizard: shallower or wider

VERSION = "3"

base_config = {
    "run.job_class": "@jobs.TrainModel",
    "TrainModel.chess_database_cls": "@data.PandasPolicyDataProvider",

    "PandasPolicyDataProvider.data_path": "/leela_generator_data/full_dataset_k=1.pkl",
    # "PandasPolicyDataProvider.eval_datapoints": 64,

    # "ResultFilter.winner_or_looser": "winner",

    "TrainModel.model_config_cls": "@transformers.BartConfig",
    "TrainModel.training_args_cls": "@transformers.TrainingArguments",

    "GlobalParamsHandler.out_dir": f"/leela_models/v{VERSION}/subgoals_k=1/medium_small_model",
    # "GlobalParamsHandler.data_location": "/leela_data_processed/full_dataset",
    "GlobalParamsHandler.learning_rate": 0.0003,
    "GlobalParamsHandler.path_format": ["learning_rate"],


    "BartConfig.vocab_size": 4600,
    "BartConfig.max_position_embeddings": 128,
    "BartConfig.encoder_layers": 8,
    "BartConfig.decoder_layers": 8,
    "BartConfig.encoder_attention_heads": 8,
    "BartConfig.decoder_attention_heads": 8,
    "BartConfig.decoder_ffn_dim": 2048,
    "BartConfig.encoder_ffn_dim": 2048,
    "BartConfig.d_model": 512,
    "BartConfig.dropout": 0.1,

    "TrainingArguments.num_train_epochs": 1,
    "TrainingArguments.per_device_train_batch_size": 900,
    "TrainingArguments.per_device_eval_batch_size": 900,
    "TrainingArguments.warmup_steps": 500,
    "TrainingArguments.weight_decay": 0.01,
    "TrainingArguments.logging_steps": 50,
    "TrainingArguments.evaluation_strategy": "steps",
    "TrainingArguments.eval_steps": 200,
    "TrainingArguments.learning_rate": 0.0002,

    "use_neptune": True,
}

params_grid = {
    "idx": [0],
    # "GlobalParamsHandler.learning_rate": [0.0001, 0.0002, 0.0003],
}

experiments_list = create_experiments_helper(
    experiment_name=f"medium-leela-subgoals_k=1-train-v{VERSION}k1",
    project_name="pmtest/subgoal-chess",
    base_config=base_config,
    params_grid=params_grid,
    script="python3 -m runner --mrunner",
    exclude=["data", ".pytest_cache", "out", ".git"],
    python_path="",
    tags=["leela", "train", "medium", "subgoals_k=1", f"v{VERSION}"],
    with_neptune=True,
    env={},
)
