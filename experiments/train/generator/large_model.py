from mrunner.helpers.specification_helper import create_experiments_helper

#Lizard: shallower or wider

VERSION = "3"

base_config = {
    "run.job_class": "@jobs.TrainModel",
    "TrainModel.chess_database_cls": "@data.PandasSubgoalDataProvider",

    "TrainModel.model_config_cls": "@transformers.BartConfig",
    "TrainModel.training_args_cls": "@transformers.TrainingArguments",

    "GlobalParamsHandler.k": 3,
    "GlobalParamsHandler.out_dir": f"/leela_models/v{VERSION}/generator/large_model",
    "GlobalParamsHandler.data_location": "/leela_generator_data/full_dataset",
    "GlobalParamsHandler.data_type": "generator",
    "GlobalParamsHandler.path_format": ["k", "learning_rate"],


    "BartConfig.vocab_size": 512,
    "BartConfig.max_position_embeddings": 128,
    "BartConfig.encoder_layers": 16,
    "BartConfig.decoder_layers": 16,
    "BartConfig.encoder_attention_heads": 16,
    "BartConfig.decoder_attention_heads": 16,
    "BartConfig.decoder_ffn_dim": 2048,
    "BartConfig.encoder_ffn_dim": 2048,
    "BartConfig.d_model": 512,
    "BartConfig.dropout": 0.01,

    "TrainingArguments.num_train_epochs": 1,
    "TrainingArguments.per_device_train_batch_size": 200,
    "TrainingArguments.per_device_eval_batch_size": 200,
    "TrainingArguments.warmup_steps": 500,
    "TrainingArguments.weight_decay": 0.01,
    "TrainingArguments.logging_steps": 50,
    "TrainingArguments.evaluation_strategy": "steps",
    "TrainingArguments.eval_steps": 200,
    "TrainingArguments.learning_rate": 0.0001,

    "use_neptune": True,
}

params_grid = {
    "idx": [0],
    "GlobalParamsHandler.k": [2,3,4,5,6],
    # "GlobalParamsHandler.learning_rate": [0.0001, 0.0002, 0.0003],
}

experiments_list = create_experiments_helper(
    experiment_name=f"lerge-leela-gen-train",
    project_name="pmtest/subgoal-chess",
    base_config=base_config,
    params_grid=params_grid,
    script="python3 -m runner --mrunner",
    exclude=["data", ".pytest_cache", "out", ".git"],
    python_path="",
    tags=["leela", "train", "medium", "generator", f"v{VERSION}"],
    with_neptune=True,
    env={},
)
