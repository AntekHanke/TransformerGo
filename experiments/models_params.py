common_train_params = {
    "from_scratch": {
        "run.job_class": "@jobs.TrainModelFromScratch",
        "TrainModelFromScratch.model_config_cls": "@transformers.BartConfig",
        "TrainModelFromScratch.training_args_cls": "@transformers.TrainingArguments",
        "TrainModelFromScratch.train_data_provider": "@data.PandasIterableSubgoalDataProvider",
        "TrainModelFromScratch.eval_data_provider": "@data.PandasStaticSubgoalDataProvider",
        "use_neptune": True,
    },
    "resume": {
        "run.job_class": "@jobs.ResumeTraining",
        "ResumeTraining.model_config_cls": "@transformers.BartConfig",
        "ResumeTraining.training_args_cls": "@transformers.TrainingArguments",
        "ResumeTraining.train_data_provider": "@data.PandasIterableSubgoalDataProvider",
        "ResumeTraining.eval_data_provider": "@data.PandasStaticSubgoalDataProvider",
        "use_neptune": True,
    },
}

ultra_small_model = {
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
    "TrainingArguments.eval_steps": 2,
    "TrainingArguments.learning_rate": 3e-5,
}

medium_model = {
    "BartConfig.vocab_size": 512,
    "BartConfig.max_position_embeddings": 128,
    "BartConfig.encoder_layers": 8,
    "BartConfig.decoder_layers": 8,
    "BartConfig.encoder_attention_heads": 8,
    "BartConfig.decoder_attention_heads": 8,
    "BartConfig.decoder_ffn_dim": 2048,
    "BartConfig.encoder_ffn_dim": 2048,
    "BartConfig.d_model": 512,
    "BartConfig.dropout": 0.1,
    "TrainingArguments.max_steps": 500000,
    "TrainingArguments.per_device_train_batch_size": 2600,
    "TrainingArguments.per_device_eval_batch_size": 2600,
    "TrainingArguments.warmup_steps": 2500,
    "TrainingArguments.weight_decay": 0.01,
    "TrainingArguments.logging_steps": 100,
    "TrainingArguments.evaluation_strategy": "steps",
    "TrainingArguments.eval_steps": 500,
    "TrainingArguments.learning_rate": 3e-5,
}
