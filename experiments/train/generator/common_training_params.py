from datetime import date

generator_global_params = {
    "GlobalParamsHandler.path_type": "generator",
    "GlobalParamsHandler.k": 3,
    "GlobalParamsHandler.path_format": ["k", "learning_rate"],
}


ultra_small_model_config = {
    "run.job_class": "@jobs.TrainModel",
    "TrainModel.train_data_provider": "@data.PandasIterableSubgoalDataProvider",
    "TrainModel.eval_data_provider": "@data.PandasStaticSubgoalDataProvider",
    "TrainModel.eval_n_batches": 4,
    "TrainModel.path_to_training_data": "/ultra_small_data/train_small.pkl",
    "TrainModel.path_to_eval_data": "/ultra_small_data/eval_small.pkl",
    "TrainModel.files_batch_size": 5,
    "TrainModel.model_config_cls": "@transformers.BartConfig",
    "TrainModel.training_args_cls": "@transformers.TrainingArguments",

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

    "use_neptune": True,
}


small_model_config = {
    "run.job_class": "@jobs.TrainModel",
    "TrainModel.train_data_provider": "@data.PandasIterableSubgoalDataProvider",
    "TrainModel.eval_data_provider": "@data.PandasStaticSubgoalDataProvider",
    "TrainModel.eval_n_batches": 4,
    "TrainModel.path_to_training_data": "/subgoals_dataset_lichess/subgoals_k=3/datapoints_train",
    "TrainModel.path_to_eval_data": "/subgoals_dataset_lichess/subgoals_k=3/datapoints_eval",
    "TrainModel.files_batch_size": 500,
    "TrainModel.model_config_cls": "@transformers.BartConfig",
    "TrainModel.training_args_cls": "@transformers.TrainingArguments",

    "BartConfig.vocab_size": 512,
    "BartConfig.max_position_embeddings": 128,
    "BartConfig.encoder_layers": 2,
    "BartConfig.decoder_layers": 2,
    "BartConfig.encoder_attention_heads": 2,
    "BartConfig.decoder_attention_heads": 2,
    "BartConfig.decoder_ffn_dim": 128,
    "BartConfig.encoder_ffn_dim": 128,
    "BartConfig.d_model": 32,
    "BartConfig.dropout": 0.1,

    "TrainingArguments.max_steps": 60000,
    "TrainingArguments.per_device_train_batch_size": 1024,
    "TrainingArguments.per_device_eval_batch_size": 1024,
    "TrainingArguments.warmup_steps": 500,
    "TrainingArguments.weight_decay": 0.01,
    "TrainingArguments.logging_steps": 50,
    "TrainingArguments.evaluation_strategy": "steps",
    "TrainingArguments.eval_steps": 200,
    "TrainingArguments.learning_rate": 3e-5,

    "use_neptune": True,
}


medium_model_config = {
    "run.job_class": "@jobs.TrainModel",
    "TrainModel.train_data_provider": "@data.PandasIterableSubgoalDataProvider",
    "TrainModel.eval_data_provider": "@data.PandasStaticSubgoalDataProvider",
    "TrainModel.eval_n_batches": 1000,
    "TrainModel.path_to_training_data": "/data/k_3",
    "TrainModel.path_to_eval_data": "/data/k_3_eval",
    "TrainModel.files_batch_size": 1000,
    "TrainModel.model_config_cls": "@transformers.BartConfig",
    "TrainModel.training_args_cls": "@transformers.TrainingArguments",

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

    "TrainingArguments.max_steps": 60000,
    "TrainingArguments.per_device_train_batch_size": 2600,
    "TrainingArguments.per_device_eval_batch_size": 2600,
    "TrainingArguments.warmup_steps": 1500,
    "TrainingArguments.weight_decay": 0.01,
    "TrainingArguments.logging_steps": 50,
    "TrainingArguments.evaluation_strategy": "steps",
    "TrainingArguments.eval_steps": 200,
    "TrainingArguments.learning_rate": 3e-5,

    "use_neptune": True,
}