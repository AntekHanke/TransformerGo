import random
from datetime import date

from mrunner.helpers.specification_helper import create_experiments_helper

base_config = {
    "run.job_class": "@jobs.TrainModel",
    "TrainModel.train_data_provider": "@data.IterableSubgoalDataLoader",
    "TrainModel.eval_data_provider": "@data.PandasStaticDataProvider",
    "TrainModel.path_to_training_data": "/ultra_small_data/lichess_elite_2015-10.pgn_train_part_0.pkl",
    "TrainModel.path_to_eval_data": "/ultra_small_data/lichess_elite_2015-10.pgn_train_part_0.pkl",
    "TrainModel.files_batch_size": 1,
    "TrainModel.model_config_cls": "@transformers.BartConfig",
    "TrainModel.training_args_cls": "@transformers.TrainingArguments",

    "GlobalParamsHandler.path_type": "generator",
    "GlobalParamsHandler.k": 3,
    "GlobalParamsHandler.out_dir": f"/out/ultra_small_generator/{date.today()}/{random.randint(0, 1000000)}/",
    "GlobalParamsHandler.path_format": ["k"],


    "BartConfig.vocab_size": 4600,
    "BartConfig.max_position_embeddings": 128,
    "BartConfig.encoder_layers": 2,
    "BartConfig.decoder_layers": 2,
    "BartConfig.encoder_attention_heads": 2,
    "BartConfig.decoder_attention_heads": 2,
    "BartConfig.decoder_ffn_dim": 64,
    "BartConfig.encoder_ffn_dim": 64,
    "BartConfig.d_model": 32,
    "BartConfig.dropout": 0.1,

    "TrainingArguments.max_steps": 5000,
    "TrainingArguments.per_device_train_batch_size": 4,
    "TrainingArguments.per_device_eval_batch_size": 8,
    "TrainingArguments.warmup_steps": 10,
    "TrainingArguments.weight_decay": 0.01,
    "TrainingArguments.logging_steps": 1,
    "TrainingArguments.evaluation_strategy": "steps",
    "TrainingArguments.eval_steps": 2,
    "TrainingArguments.learning_rate": 3e-5,
    "TrainingArguments.eval_accumulation_steps": 2,
    "use_neptune": True,
}

params_grid = {
    "idx": [0],
    "GlobalParamsHandler.k": [3],
    "GlobalParamsHandler.learning_rate": [3e-5],
}

experiments_list = create_experiments_helper(
    experiment_name=f"ultra_small_generator-{base_config['GlobalParamsHandler.out_dir']}",
    project_name="pmtest/subgoal-chess",
    base_config=base_config,
    params_grid=params_grid,
    script="python3 -m runner --mrunner",
    exclude=["data", ".pytest_cache", "out", ".git"],
    python_path="",
    tags=["chessengines games", "train", "small", "subgoals", "k=3"],
    with_neptune=True,
    env={},
)
