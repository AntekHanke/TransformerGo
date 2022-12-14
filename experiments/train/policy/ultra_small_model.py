from mrunner.helpers.specification_helper import create_experiments_helper

#Lizard: shallower or wider

VERSION = "3"

base_config = {
    "run.job_class": "@jobs.TrainModel",
    "TrainModel.chess_database_cls": "@data.PandasPolicyDataProvider",

    "PandasPolicyDataProvider.data_path": "/leela_generator_data/prom_full_dataset_k=4.pkl",
    "PandasPolicyDataProvider.eval_datapoints": 64,

    # "ResultFilter.winner_or_looser": "winner",

    "TrainModel.model_config_cls": "@transformers.BartConfig",
    "TrainModel.training_args_cls": "@transformers.TrainingArguments",

    "GlobalParamsHandler.out_dir": "/leela_models/v1/policy/ultra_small_model",
    # "GlobalParamsHandler.data_location": "/leela_data_processed/full_dataset",
    "GlobalParamsHandler.learning_rate": 0.0003,
    "GlobalParamsHandler.path_format": ["learning_rate"],

    "BartConfig.vocab_size": 4600,
    "BartConfig.max_position_embeddings": 128,
    "BartConfig.encoder_layers": 2,
    "BartConfig.decoder_layers": 2,
    "BartConfig.encoder_attention_heads": 2,
    "BartConfig.decoder_attention_heads": 2,
    "BartConfig.decoder_ffn_dim": 128,
    "BartConfig.encoder_ffn_dim": 128,
    "BartConfig.d_model": 32,
    "BartConfig.dropout": 0.05,

    "TrainingArguments.num_train_epochs": 1,
    "TrainingArguments.per_device_train_batch_size": 2,
    "TrainingArguments.per_device_eval_batch_size": 2,
    "TrainingArguments.warmup_steps": 500,
    "TrainingArguments.weight_decay": 0.01,
    "TrainingArguments.logging_steps": 50,
    "TrainingArguments.evaluation_strategy": "steps",
    "TrainingArguments.eval_steps": 5,
    "TrainingArguments.learning_rate": 0.0002,

    "use_neptune": True,
}

params_grid = {
    "idx": [0],
    # "GlobalParamsHandler.k": [3],
    # "GlobalParamsHandler.learning_rate": [0.0001, 0.0002, 0.0003, 0.001],
}

experiments_list = create_experiments_helper(
    experiment_name=f"ultra-small-leela-policy-train",
    project_name="pmtest/subgoal-chess",
    base_config=base_config,
    params_grid=params_grid,
    script="python3 -m runner --mrunner",
    exclude=["data", ".pytest_cache", "out", ".git"],
    python_path="",
    tags=["leela", "train", "small"],
    with_neptune=True,
    env={},
)
