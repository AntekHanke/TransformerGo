from datetime import date

from mrunner.helpers.specification_helper import create_experiments_helper


base_config = {
    "run.job_class": "@jobs.TrainModel",
    "TrainModel.iterable_dataset_class": "@data.IterableSubgoalDataLoader",

    "TrainModel.path_to_training_data": None,
    "TrainModel.path_to_eval_data": None,

    "TrainModel.model_config_cls": "@transformers.BartConfig",
    "TrainModel.training_args_cls": "@transformers.TrainingArguments",

    "GlobalParamsHandler.path_type": "generator",
    "GlobalParamsHandler.k": 1,
    "GlobalParamsHandler.data_location": "/chess_data",
    "GlobalParamsHandler.out_dir": f"/leela_models/generator/ultra_small_model/{date.today()}",
    "GlobalParamsHandler.path_format": ["k", "learning_rate"],

    "BartConfig.vocab_size": 4562,
    "BartConfig.max_position_embeddings": 90,
    "BartConfig.encoder_layers": 2,
    "BartConfig.decoder_layers": 2,
    "BartConfig.encoder_attention_heads": 2,
    "BartConfig.decoder_attention_heads": 2,
    "BartConfig.decoder_ffn_dim": 128,
    "BartConfig.encoder_ffn_dim": 128,
    "BartConfig.d_model": 32,
    "BartConfig.dropout": 0.05,

    "TrainingArguments.max_steps": 100,
    "TrainingArguments.per_device_train_batch_size": 128,
    "TrainingArguments.per_device_eval_batch_size": 128,
    "TrainingArguments.warmup_steps": 500,
    "TrainingArguments.weight_decay": 0.01,
    "TrainingArguments.logging_steps": 2,
    "TrainingArguments.evaluation_strategy": "steps",
    "TrainingArguments.eval_steps": 200,
    "TrainingArguments.learning_rate": 3e-5,

    "use_neptune": True,
}

params_grid = {
    "idx": [0],
    "GlobalParamsHandler.k": [1],
    "GlobalParamsHandler.learning_rate": [3e-5],
}

experiments_list = create_experiments_helper(
    experiment_name=f"ultra-small-leela-subgoals_k=1 - test",
    project_name="pmtest/subgoal-chess",
    base_config=base_config,
    params_grid=params_grid,
    script="python3 -m runner --mrunner",
    exclude=["data", ".pytest_cache", "out", ".git"],
    python_path="",
    tags=["leela", "train", "small", "test", "subgoals_k=1"],
    with_neptune=True,
    env={},
)
