from mrunner.helpers.specification_helper import create_experiments_helper


base_config = {
    "run.job_class": "@jobs.TrainModel",
    "TrainModel.chess_database_cls": "@data.PandasSubgoalDataProvider",
    "TrainModel.paths_provider_cls": "@data.TrainOnLeelaPathsProvider",
    "TrainModel.model_config_cls": "@transformers.BartConfig",
    "TrainModel.training_args_cls": "@transformers.TrainingArguments",
    "PandasSubgoalDataProvider.paths_provider_cls": "@data.TrainOnLeelaPathsProvider",
    "TrainOnLeelaPathsProvider.pickle_df_path": f"/save_data/full_dataset",
    "TrainOnLeelaPathsProvider.save_models_to": "/leela_models/medium",
    "TrainOnLeelaPathsProvider.k": 6,
    "BartConfig.vocab_size": 512,
    "BartConfig.max_position_embeddings": 128,
    "BartConfig.encoder_layers": 12,
    "BartConfig.decoder_layers": 12,
    "BartConfig.encoder_attention_heads": 16,
    "BartConfig.decoder_attention_heads": 16,
    "BartConfig.decoder_ffn_dim": 1024,
    "BartConfig.encoder_ffn_dim": 1024,
    "BartConfig.d_model": 512,
    "BartConfig.dropout": 0.01,
    "TrainingArguments.num_train_epochs": 1,
    "TrainingArguments.per_device_train_batch_size": 256,
    "TrainingArguments.per_device_eval_batch_size": 256,
    "TrainingArguments.warmup_steps": 500,
    "TrainingArguments.weight_decay": 0.01,
    "TrainingArguments.logging_steps": 50,
    "TrainingArguments.evaluation_strategy": "steps",
    "TrainingArguments.eval_steps": 200,
    "TrainingArguments.learning_rate": 0.0003,
    "use_neptune": True,
}

params_grid = {"idx": [0], "TrainOnLeelaPathsProvider.k": [1, 2, 3, 4]}

experiments_list = create_experiments_helper(
    experiment_name=f"2-medium-leela-gen-train",
    project_name="pmtest/subgoal-chess",
    base_config=base_config,
    params_grid=params_grid,
    script="python3 -m runner --mrunner",
    exclude=["data", ".pytest_cache", "out", ".git"],
    python_path="",
    tags=["leela", "train", "medium"],
    with_neptune=True,
    env={},
)
