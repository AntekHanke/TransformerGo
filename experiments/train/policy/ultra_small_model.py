from datetime import date
from mrunner.helpers.specification_helper import create_experiments_helper

base_config = {
    "run.job_class": "@jobs.TrainModel",
    "TrainModel.iterable_dataset_class": "@data.IterablePolicyDataLoader",
    "TrainModel.path_to_training_data": "/leela_generator_data_train/subgoals_k=1",
    "TrainModel.path_to_eval_data": "/leela_generator_data_eval/subgoals_k=1",
    "TrainModel.files_batch_size": 2,
    "TrainModel.model_config_cls": "@transformers.BartConfig",
    "TrainModel.training_args_cls": "@transformers.TrainingArguments",
    "GlobalParamsHandler.out_dir": f"/leela_models/subgoals_k=1/ultra_small_model/{date.today()}",
    "GlobalParamsHandler.path_type": "raw_path",
    "BartConfig.vocab_size": 4600,
    "BartConfig.max_position_embeddings": 100,
    "BartConfig.encoder_layers": 2,
    "BartConfig.decoder_layers": 2,
    "BartConfig.encoder_attention_heads": 2,
    "BartConfig.decoder_attention_heads": 2,
    "BartConfig.decoder_ffn_dim": 128,
    "BartConfig.encoder_ffn_dim": 128,
    "BartConfig.d_model": 32,
    "BartConfig.dropout": 0.05,
    "TrainingArguments.max_steps": 5000,
    "TrainingArguments.per_device_train_batch_size": 512,
    "TrainingArguments.per_device_eval_batch_size": 512,
    "TrainingArguments.warmup_steps": 50,
    "TrainingArguments.weight_decay": 0.01,
    "TrainingArguments.logging_steps": 2,
    "TrainingArguments.evaluation_strategy": "steps",
    "TrainingArguments.eval_steps": 200,
    "TrainingArguments.learning_rate": 0.0002,
    "use_neptune": True,
}

params_grid = {
    "idx": [0],
}

experiments_list = create_experiments_helper(
    experiment_name=f"ultra-small-leela-subgoals_k=1",
    project_name="pmtest/subgoal-chess",
    base_config=base_config,
    params_grid=params_grid,
    script="python3 -m runner --mrunner",
    exclude=["data", ".pytest_cache", "out", ".git"],
    python_path="",
    tags=["leela", "train", "ultra-small", "subgoals_k=1"],
    with_neptune=True,
    env={},
)
