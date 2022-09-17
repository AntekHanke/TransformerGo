from mrunner.helpers.specification_helper import create_experiments_helper

k = 3

base_config = {
    "run.job_class": "@jobs.TrainModel",
    "TrainModel.chess_database": "@data.ChessCLLPDataGenerator",

    "ChessCLLPDataGenerator.k": k,
    "ChessCLLPDataGenerator.pgn_file": "/home/plgrid/plgtodrzygozdz/subgoal_chess/database.pgn",
    "ChessCLLPDataGenerator.filter": "@filters.NoFilter",
    "ChessCLLPDataGenerator.p_sample": 0.1,
    "ChessCLLPDataGenerator.n_data": 5*10**5,
    "ChessCLLPDataGenerator.log_samples_limit": 100,
    "ChessCLLPDataGenerator.p_log_sample": 0.01,

    "TrainModel.save_model_path": f"/home/plgrid/plgtodrzygozdz/chess_models/cllp_k={k}/",

    "BartConfig.vocab_size": 512,
    "BartConfig.max_position_embeddings": 128,
    "BartConfig.encoder_layers": 12,
    "BartConfig.decoder_layers": 12,
    "BartConfig.encoder_attention_heads": 16,
    "BartConfig.decoder_attention_heads": 16,
    "BartConfig.decoder_ffn_dim": 1024,
    "BartConfig.encoder_ffn_dim": 1024,
    "BartConfig.d_model": 512,
    "BartConfig.dropout": 0.01,

    "TrainingArguments.num_train_epochs": 1,
    "TrainingArguments.per_device_train_batch_size": 256,
    "TrainingArguments.per_device_eval_batch_size": 256,
    "TrainingArguments.warmup_steps": 500,
    "TrainingArguments.weight_decay": 0.01,
    "TrainingArguments.logging_steps": 50,
    "TrainingArguments.evaluation_strategy": "steps",
    "TrainingArguments.eval_steps": 200,
    "TrainingArguments.learning_rate": 0.0001,
}

params_grid = {
    "idx": [0],
}

experiments_list = create_experiments_helper(
    experiment_name=f"CLLP-train_k={k}",
    project_name="pmtest/subgoal-chess",
    base_config=base_config,
    params_grid=params_grid,
    script="python3 -m runner --mrunner",
    exclude=["data", ".pytest_cache", "out", ".git"],
    python_path="",
    tags=["CLLP", "train", f"k={k}"],
    with_neptune=True,
    env={},
)
