import random
from datetime import date
from mrunner.helpers.specification_helper import create_experiments_helper

batch_size = {"ares": 1024, "athena": 600}

base_config = {
    "run.job_class": "@jobs.TrainModel",
    "TrainModel.iterable_dataset_class": "@data.IterablePolicyDataLoader",
    "TrainModel.files_batch_size": 10,
    "TrainModel.path_to_training_data": "/save_data/policy_data/train",
    "TrainModel.path_to_eval_data": "/save_data/policy_data/eval",
    "TrainModel.model_config_cls": "@transformers.BartConfig",
    "TrainModel.training_args_cls": "@transformers.TrainingArguments",
    "GlobalParamsHandler.out_dir": f"/leela_models/policy/medium_large_model/{date.today()}/{random.randint(0, 100000)}",
    "GlobalParamsHandler.path_type": "raw_path",
    "BartConfig.vocab_size": 4600,
    "BartConfig.max_position_embeddings": 100,
    "BartConfig.encoder_layers": 8,
    "BartConfig.decoder_layers": 8,
    "BartConfig.encoder_attention_heads": 16,
    "BartConfig.decoder_attention_heads": 16,
    "BartConfig.decoder_ffn_dim": 4096,
    "BartConfig.encoder_ffn_dim": 4096,
    "BartConfig.d_model": 1024,
    "BartConfig.dropout": 0.1,
    "TrainingArguments.max_steps": 50000,
    "TrainingArguments.per_device_train_batch_size": batch_size["athena"],
    "TrainingArguments.per_device_eval_batch_size": batch_size["athena"],
    "TrainingArguments.warmup_steps": 1500,
    "TrainingArguments.weight_decay": 0.01,
    "TrainingArguments.logging_steps": 50,
    "TrainingArguments.evaluation_strategy": "steps",
    "TrainingArguments.eval_steps": 200,
    "TrainingArguments.learning_rate": 3e-4,
    "use_neptune": True,
}

params_grid = {"idx": [0]}

experiments_list = create_experiments_helper(
    experiment_name=f"{base_config['TrainingArguments.learning_rate']}-lr-large-policy",
    project_name="pmtest/subgoal-chess",
    base_config=base_config,
    params_grid=params_grid,
    script="python3 -m runner --mrunner",
    exclude=["data", ".pytest_cache", "out", ".git"],
    python_path="",
    tags=["leela", "train", "large", "subgoals_k=1"],
    with_neptune=True,
    env={},
)
