import random
from datetime import date

from mrunner.helpers.specification_helper import create_experiments_helper

base_config = {
    "run.job_class": "@jobs.TrainModel",
    "TrainModel.train_data_provider": "@data.PandasIterableSubgoalDataProvider",
    "TrainModel.eval_data_provider": "@data.PandasStaticSubgoalDataProvider",
    "TrainModel.eval_n_batches": 4,
    "TrainModel.path_to_training_data": "/ultra_small_data/train_small.pkl",
    "TrainModel.path_to_eval_data": "/ultra_small_data/eval_small.pkl",

    "TrainModel.checkpoint_to_resume": "/out_models/ultra_small_generator/2023-02-14/573635/_k=3/out/checkpoint-200",

    "TrainModel.files_batch_size": 5,
    "TrainModel.model_config_cls": "@transformers.BartConfig",
    "TrainModel.training_args_cls": "@transformers.TrainingArguments",

    "GlobalParamsHandler.path_type": "generator",
    "GlobalParamsHandler.k": 3,
    "GlobalParamsHandler.out_dir": f"/out_models/ultra_small_generator_resume/{date.today()}/{random.randint(0, 1000000)}/",
    "GlobalParamsHandler.path_format": ["k"],

    "TrainingArguments.max_steps": 5000,
    "TrainingArguments.per_device_train_batch_size": 4,
    "TrainingArguments.per_device_eval_batch_size": 8,
    "TrainingArguments.warmup_steps": 10,
    "TrainingArguments.weight_decay": 0.01,
    "TrainingArguments.logging_steps": 1,
    "TrainingArguments.evaluation_strategy": "steps",
    "TrainingArguments.eval_steps": 2,
    "TrainingArguments.save_strategy": "steps",
    "TrainingArguments.save_steps": 100,
    "TrainingArguments.learning_rate": 3e-5,

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
