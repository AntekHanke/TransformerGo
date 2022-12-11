from mrunner.helpers.specification_helper import create_experiments_helper


base_config = {
    "run.job_class": "@jobs.TrainModel",
    "TrainModel.chess_database_cls": "@data.PolicyGamesDataGenerator",

    "PolicyGamesDataGenerator.pgn_file": "/pgn/solved_games.pgn",
    "PolicyGamesDataGenerator.chess_filter": "@filters.NoFilter",
    "PolicyGamesDataGenerator.n_data": 10**4,
    "PolicyGamesDataGenerator.log_samples_limit": 0.1,
    "PolicyGamesDataGenerator.p_sample": 0.05,


    # "ResultFilter.winner_or_looser": "winner",

    "TrainModel.model_config_cls": "@transformers.BartConfig",
    "TrainModel.training_args_cls": "@transformers.TrainingArguments",

    "GlobalParamsHandler.out_dir": "/leela_models/v1/pgn_policy/small_model",
    # "GlobalParamsHandler.data_location": "/leela_data_processed/full_dataset",
    "GlobalParamsHandler.learning_rate": 0.0003,
    "GlobalParamsHandler.path_format": ["learning_rate"],

    "BartConfig.vocab_size": 512,
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
    "TrainingArguments.per_device_train_batch_size": 32,
    "TrainingArguments.per_device_eval_batch_size": 32,
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
    experiment_name=f"ultra-small-leela-pgn_policy-train",
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
