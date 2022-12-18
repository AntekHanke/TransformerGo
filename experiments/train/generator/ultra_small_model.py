from datetime import date

from mrunner.helpers.specification_helper import create_experiments_helper

from metric_logging import source_files_register

import os

source_files_register.register(os.path.dirname(os.path.abspath('__file__')))
print(f"__file__ = {os.path.dirname(os.path.abspath('__file__'))}")


base_config = {
    "run.job_class": "@jobs.TrainModel",
    "TrainModel.iterable_dataset_class": "@data.IterableSubgoalDataLoader",

    "TrainModel.path_to_training_data": "/leela_generator_data/train/generator",
    "TrainModel.path_to_eval_data": "/leela_generator_data/eval",

    "TrainModel.model_config_cls": "@transformers.BartConfig",
    "TrainModel.training_args_cls": "@transformers.TrainingArguments",

    "GlobalParamsHandler.out_dir": f"/leela_models/test_training_generator/leela_models/generator/ultra_small_model/{date.today()}",
    "GlobalParamsHandler.path_type": "full_info",
    "GlobalParamsHandler.path_format": ["learning_rate"],

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
    "TrainingArguments.learning_rate": 0.0002,

    "use_neptune": True,
}

params_grid = {
    "idx": [0],
    "GlobalParamsHandler.learning_rate": [0.0001],
}

experiments_list = create_experiments_helper(
    experiment_name=f"ultra-small-leela-policy",
    project_name="pmtest/subgoal-chess",
    base_config=base_config,
    params_grid=params_grid,
    script="python3 -m runner --mrunner",
    exclude=["data", ".pytest_cache", "out", ".git"],
    python_path="",
    tags=["leela", "train", "small", "test", "policy"],
    with_neptune=True,
    env={},
)
