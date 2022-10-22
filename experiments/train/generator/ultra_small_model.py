from mrunner.helpers.specification_helper import create_experiments_helper


base_config = {
    "run.job_class": "@jobs.TrainModel",
    "TrainModel.chess_database_cls": "@data.PandasSubgoalDataProvider",

    # "TrainModel.global_params_handler": "@params.GlobalParamsHandler",
    "TrainModel.model_config_cls": "@transformers.BartConfig",
    "TrainModel.training_args_cls": "@transformers.TrainingArguments",

    "GlobalParamsHandler.k": 3,
    "GlobalParamsHandler.out_dir": "experiments/train/generator/small_model",
    "GlobalParamsHandler.data_location": "/leela_data_processed/full_dataset",


    "BartConfig.vocab_size": 512,
    "BartConfig.max_position_embeddings": 128,
    "BartConfig.encoder_layers": 4,
    "BartConfig.decoder_layers": 4,
    "BartConfig.encoder_attention_heads": 4,
    "BartConfig.decoder_attention_heads": 4,
    "BartConfig.decoder_ffn_dim": 256,
    "BartConfig.encoder_ffn_dim": 256,
    "BartConfig.d_model": 64,
    "BartConfig.dropout": 0.1,

    "TrainingArguments.num_train_epochs": 1,
    "TrainingArguments.per_device_train_batch_size": 512,
    "TrainingArguments.per_device_eval_batch_size": 512,
    "TrainingArguments.warmup_steps": 100,
    "TrainingArguments.weight_decay": 0.01,
    "TrainingArguments.logging_steps": 50,
    "TrainingArguments.evaluation_strategy": "steps",
    "TrainingArguments.eval_steps": 200,
    "TrainingArguments.learning_rate": 0.0002,

    "use_neptune": False,
}

params_grid = {
    "idx": [0],
    # "GlobalParamsHandler.k": [3]
}

experiments_list = create_experiments_helper(
    experiment_name=f"small-leela-gen-train",
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
