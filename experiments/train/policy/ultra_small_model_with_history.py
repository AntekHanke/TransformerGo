from datetime import date
from mrunner.helpers.specification_helper import create_experiments_helper

base_config = {
    "run.job_class": "@jobs.TrainModel",

    "TrainModel.iterable_dataset_class": "@data.IterablePolicyDataWithHistoryLoader",
    "TrainModel.path_to_training_data": "/ultra_small_data/train_small.pkl",
    "TrainModel.path_to_eval_data": "/ultra_small_data/eval_small.pkl",
    "TrainModel.files_batch_size": 1,
    "TrainModel.model_config_cls": "@transformers.BartConfig",
    "TrainModel.training_args_cls": "@transformers.TrainingArguments",
    "TrainModel.out_dir": "/out/policy/ultra_small/added_history",

    "BartConfig.vocab_size": 4600,
    "BartConfig.max_position_embeddings": 300,
    "BartConfig.encoder_layers": 4,
    "BartConfig.decoder_layers": 4,
    "BartConfig.encoder_attention_heads": 2,
    "BartConfig.decoder_attention_heads": 2,
    "BartConfig.decoder_ffn_dim": 128,
    "BartConfig.encoder_ffn_dim": 128,
    "BartConfig.d_model": 32,
    "BartConfig.dropout": 0.05,

    "TrainingArguments.max_steps": 5000,
    "TrainingArguments.per_device_train_batch_size": 1,
    "TrainingArguments.per_device_eval_batch_size": 1,
    "TrainingArguments.warmup_steps": 10,
    "TrainingArguments.weight_decay": 0.01,
    "TrainingArguments.logging_steps": 2,
    "TrainingArguments.evaluation_strategy": "steps",
    "TrainingArguments.eval_steps": 20,
    "TrainingArguments.learning_rate": 2e-4,

    "use_neptune": True,
}

params_grid = {
    "idx": [0],
}

experiments_list = create_experiments_helper(
    experiment_name=f"ultra-small-policy-model-with-history",
    project_name="pmtest/subgoal-chess",
    base_config=base_config,
    params_grid=params_grid,
    script="python3 -m runner --mrunner",
    exclude=["data", ".pytest_cache", "out", ".git"],
    python_path="",
    tags=["policy", "train", "ultra-small", "with-history"],
    with_neptune=True,
    env={},
)
