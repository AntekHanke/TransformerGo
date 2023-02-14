from datetime import date


ultra_small_model_config = {
    "run.job_class": "@jobs.TrainModel",
    "TrainModel.eval_n_batches": 4,
    "TrainModel.path_to_training_data": "/ultra_small_data/train_small.pkl",
    "TrainModel.path_to_eval_data": "/ultra_small_data/eval_small.pkl",
    "TrainModel.files_batch_size": 1,
    "TrainModel.model_config_cls": "@transformers.BartConfig",
    "TrainModel.training_args_cls": "@transformers.TrainingArguments",

    "BartConfig.vocab_size": 4600,
    "BartConfig.max_position_embeddings": 100,
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

medium_model_config = {
    "run.job_class": "@jobs.TrainModel",
    "TrainModel.eval_n_batches": 1000,
    "TrainModel.path_to_training_data": "/data/k_3",
    "TrainModel.path_to_eval_data": "/data/k_3_eval",
    "TrainModel.files_batch_size": 1000,
    "TrainModel.model_config_cls": "@transformers.BartConfig",
    "TrainModel.training_args_cls": "@transformers.TrainingArguments",

    "BartConfig.vocab_size": 4600,
    "BartConfig.max_position_embeddings": 100,
    "BartConfig.encoder_layers": 8,
    "BartConfig.decoder_layers": 8,
    "BartConfig.encoder_attention_heads": 8,
    "BartConfig.decoder_attention_heads": 8,
    "BartConfig.decoder_ffn_dim": 2048,
    "BartConfig.encoder_ffn_dim": 2048,
    "BartConfig.d_model": 512,
    "BartConfig.dropout": 0.1,

    "TrainingArguments.max_steps": 60000,
    "TrainingArguments.per_device_train_batch_size": 5700,
    "TrainingArguments.per_device_eval_batch_size": 5700,
    "TrainingArguments.warmup_steps": 1500,
    "TrainingArguments.weight_decay": 0.01,
    "TrainingArguments.logging_steps": 50,
    "TrainingArguments.evaluation_strategy": "steps",
    "TrainingArguments.eval_steps": 200,
    "TrainingArguments.learning_rate": 3e-5,

    "use_neptune": True,
}
