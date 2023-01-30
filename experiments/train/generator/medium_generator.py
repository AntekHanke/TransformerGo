from datetime import date

from mrunner.helpers.specification_helper import create_experiments_helper

base_config = {
    "run.job_class": "@jobs.TrainModel",
    "TrainModel.iterable_dataset_class": "@data.IterableSubgoalDataLoader",
    "TrainModel.path_to_training_data": None,
    "TrainModel.path_to_eval_data": None,
    "TrainModel.files_batch_size": 25,
    "TrainModel.model_config_cls": "@transformers.BartConfig",
    "TrainModel.training_args_cls": "@transformers.TrainingArguments",
    "GlobalParamsHandler.path_type": "generator",
    "GlobalParamsHandler.k": 1,
    "GlobalParamsHandler.data_location": "/chess_data",
    "GlobalParamsHandler.out_dir": f"/leela_models/generator/medium_model/{date.today()}",
    "GlobalParamsHandler.path_format": ["k", "learning_rate"],
    "BartConfig.vocab_size": 512,
    "BartConfig.max_position_embeddings": 128,
    "BartConfig.encoder_layers": 8,
    "BartConfig.decoder_layers": 8,
    "BartConfig.encoder_attention_heads": 8,
    "BartConfig.decoder_attention_heads": 8,
    "BartConfig.decoder_ffn_dim": 2048,
    "BartConfig.encoder_ffn_dim": 2048,
    "BartConfig.d_model": 512,
    "BartConfig.dropout": 0.1,
    "TrainingArguments.max_steps": 60000,
    "TrainingArguments.per_device_train_batch_size": 1100,
    "TrainingArguments.per_device_eval_batch_size": 1100,
    "TrainingArguments.warmup_steps": 1500,
    "TrainingArguments.weight_decay": 0.01,
    "TrainingArguments.logging_steps": 50,
    "TrainingArguments.evaluation_strategy": "steps",
    "TrainingArguments.eval_steps": 200,
    "TrainingArguments.learning_rate": 3e-5,
    "use_neptune": True,
}

params_grid = {
    "idx": [0],
    "GlobalParamsHandler.k": [1, 2, 3, 4, 5, 6],
    "GlobalParamsHandler.learning_rate": [3e-5],
}

experiments_list = create_experiments_helper(
    experiment_name=f"-training-medium-generator-leela-subgoals-batch_1100",
    project_name="pmtest/subgoal-chess",
    base_config=base_config,
    params_grid=params_grid,
    script="python3 -m runner --mrunner",
    exclude=["data", ".pytest_cache", "out", ".git"],
    python_path="",
    tags=["leela", "train", "medium", "subgoals", "batch_1100"],
    with_neptune=True,
    env={},
)
