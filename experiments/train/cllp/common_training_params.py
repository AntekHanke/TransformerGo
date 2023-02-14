medium_model_config = {
    "run.job_class": "@jobs.TrainModel",

    "TrainModel.train_data_provider": "@data.PandasIterableCLLPDataProvider",
    "TrainModel.eval_data_provider": "@data.PandasStaticCLLPDataProvider",
    "TrainModel.eval_n_batches": 4,
    "TrainModel.files_batch_size": 10,
    "TrainModel.path_to_eval_data": "/subgoal_chess_data/cllp_eval",
    "TrainModel.model_config_cls": "@transformers.BartConfig",
    "TrainModel.training_args_cls": "@transformers.TrainingArguments",

    "GlobalParamsHandler.path_type": "raw_path",

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
    "TrainingArguments.warmup_steps": 1500,
    "TrainingArguments.weight_decay": 0.01,
    "TrainingArguments.logging_steps": 50,
    "TrainingArguments.evaluation_strategy": "steps",
    "TrainingArguments.eval_steps": 200,
    "TrainingArguments.learning_rate": 2e-4,

    "use_neptune": True,
}