import random
from datetime import date
from mrunner.helpers.specification_helper import create_experiments_helper

batch_size = {"ares": 1024, "athena": 700, "athena_2_gpu": 1350, "athena_4_gpu": 2700}

MACHINE = "athena_4_gpu"

base_config = {
    "run.job_class": "@jobs.TrainModel",
    "TrainModel.train_data_provider": "@data.PandasIterableCLLPDataProvider",
    "TrainModel.eval_data_provider": "@data.PandasStaticCLLPDataProvider",
    "TrainModel.eval_n_batches": 4,
    "TrainModel.files_batch_size": 10,
    "TrainModel.path_to_training_data": "/subgoal_chess_data/cllp_data",
    "TrainModel.path_to_eval_data": "/subgoal_chess_data/cllp_eval",
    "TrainModel.model_config_cls": "@transformers.BartConfig",
    "TrainModel.training_args_cls": "@transformers.TrainingArguments",
    "GlobalParamsHandler.out_dir": f"/leela_models/cllp/medium_model/{date.today()}/{random.randint(0, 100000)}",
    "GlobalParamsHandler.path_type": "raw_path",
    # "GlobalParamsHandler.path_format": ['learning_rate'],
    "BartConfig.vocab_size": 4600,
    "BartConfig.max_position_embeddings": 160,
    "BartConfig.encoder_layers": 8,
    "BartConfig.decoder_layers": 8,
    "BartConfig.encoder_attention_heads": 8,
    "BartConfig.decoder_attention_heads": 8,
    "BartConfig.decoder_ffn_dim": 2048,
    "BartConfig.encoder_ffn_dim": 2048,
    "BartConfig.d_model": 512,
    "BartConfig.dropout": 0.0,
    "TrainingArguments.max_steps": 60000,
    "TrainingArguments.per_device_train_batch_size": batch_size[MACHINE],
    "TrainingArguments.per_device_eval_batch_size": batch_size[MACHINE],
    "TrainingArguments.warmup_steps": 1500,
    "TrainingArguments.weight_decay": 0.01,
    "TrainingArguments.logging_steps": 50,
    "TrainingArguments.evaluation_strategy": "steps",
    "TrainingArguments.eval_steps": 200,
    "TrainingArguments.learning_rate": 2e-4,
    "use_neptune": True,
}

params_grid = {
    "idx": [0],
    # "GlobalParamsHandler.learning_rate": [3e-5, 3e-4]
}

experiments_list = create_experiments_helper(
    experiment_name=f"EQ-b{batch_size[MACHINE]}-{base_config['TrainingArguments.learning_rate']}-lr-2g-medium-cllp",
    project_name="pmtest/subgoal-chess",
    base_config=base_config,
    params_grid=params_grid,
    script="python3 -m runner --mrunner",
    exclude=["data", ".pytest_cache", "out", ".git"],
    python_path="",
    tags=["leela", "train", "medium", "cllp"],
    with_neptune=True,
    env={},
)
