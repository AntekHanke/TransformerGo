from datetime import date

from mrunner.helpers.specification_helper import create_experiments_helper

base_config = {
    "run.job_class": "@jobs.TrainModel",

    "TrainModel.train_data_provider": "@data.PandasIterableSubgoalDataProvider",
    "TrainModel.eval_data_provider": "@data.PandasStaticSubgoalDataProvider",
    "TrainModel.eval_n_batches": 4,
    "TrainModel.path_to_training_data": "/data/k_3",
    "TrainModel.path_to_eval_data": "/data/k_3_eval",
    "TrainModel.files_batch_size": 1000,
    "TrainModel.model_config_cls": "@transformers.BartConfig",
    "TrainModel.training_args_cls": "@transformers.TrainingArguments",

    "GlobalParamsHandler.path_type": "generator",
    "GlobalParamsHandler.k": 3,
    "GlobalParamsHandler.out_dir": f"/models/generator/medium_model/{date.today()}",
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
    "TrainingArguments.per_device_train_batch_size": 400,
    "TrainingArguments.per_device_eval_batch_size": 400,
    "TrainingArguments.warmup_steps": 1500,
    "TrainingArguments.weight_decay": 0.01,
    "TrainingArguments.logging_steps": 5,
    "TrainingArguments.evaluation_strategy": "steps",
    "TrainingArguments.eval_steps": 10,
    "TrainingArguments.learning_rate": 3e-5,

    "use_neptune": True,
}

params_grid = {
    # "GlobalParamsHandler.k": [1, 2, 3, 4, 5, 6],
    "GlobalParamsHandler.learning_rate": [3e-5],
}

batch = base_config["TrainingArguments.per_device_train_batch_size"]

experiments_list = create_experiments_helper(
    experiment_name=f"-training-medium-generator_b={batch}",
    project_name="pmtest/subgoal-chess",
    base_config=base_config,
    params_grid=params_grid,
    script="python3 -m runner --mrunner",
    exclude=["data", ".pytest_cache", "out", ".git"],
    python_path="",
    tags=["leela", "train", "medium", "subgoals", "batch"],
    with_neptune=True,
    env={},
)
