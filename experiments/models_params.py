common_train_params = {
    "from_scratch": {
        "run.job_class": "@jobs.TrainModelFromScratch",
        "TrainModelFromScratch.model_config_cls": "@transformers.BartConfig",
        "TrainModelFromScratch.training_args_cls": "@transformers.TrainingArguments",
        "use_neptune": True,
    },
    "from_scratch_go_policy" : {
        "run.job_class" : "@jobs.TrainConvolutionFromScratch",
        "TrainConvolutionFromScratch.model_config_cls" : "@AlphaZero.AlphaZeroPolicyConfig",
        "TrainConvolutionFromScratch.training_args_cls" : "@transformers.TrainingArguments",
        "use_neptune" : True
    },
    "from_scratch_with_all_subgoals": {
        "run.job_class": "@jobs.TrainModelFromScratch",
        "TrainModelFromScratch.model_config_cls": "@transformers.BartConfig",
        "TrainModelFromScratch.training_args_cls": "@transformers.TrainingArguments",
        "use_neptune": True,
    },
    "resume": {
        "run.job_class": "@jobs.ResumeTraining",
        "ResumeTraining.model_config_cls": "@transformers.BartConfig",
        "ResumeTraining.training_args_cls": "@transformers.TrainingArguments",
        "use_neptune": True,
    },
    "resume_with_all_subgoals": {
        "run.job_class": "@jobs.ResumeTraining",
        "ResumeTraining.model_config_cls": "@transformers.BartConfig",
        "ResumeTraining.training_args_cls": "@transformers.TrainingArguments",
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
    "TrainingArguments.per_device_eval_batch_size": 4,
    "TrainingArguments.warmup_steps": 10,
    "TrainingArguments.weight_decay": 0.01,
    "TrainingArguments.logging_steps": 1,
    "TrainingArguments.evaluation_strategy": "steps",
    "TrainingArguments.eval_steps": 2,
    "TrainingArguments.learning_rate": 3e-5,
}

small_model = {
    "BartConfig.vocab_size": 4600,
    "BartConfig.max_position_embeddings": 128,
    "BartConfig.encoder_layers": 4,
    "BartConfig.decoder_layers": 4,
    "BartConfig.encoder_attention_heads": 4,
    "BartConfig.decoder_attention_heads": 4,
    "BartConfig.decoder_ffn_dim": 1024,
    "BartConfig.encoder_ffn_dim": 1024,
    "BartConfig.d_model": 256,
    "BartConfig.dropout": 0.1,
    "TrainingArguments.max_steps": 500000,
    "TrainingArguments.per_device_train_batch_size": 18000,
    "TrainingArguments.per_device_eval_batch_size": 18000,
    "TrainingArguments.warmup_steps": 2500,
    "TrainingArguments.weight_decay": 0.01,
    "TrainingArguments.logging_steps": 100,
    "TrainingArguments.evaluation_strategy": "steps",
    "TrainingArguments.eval_steps": 500,
    "TrainingArguments.learning_rate": 2e-4,
}

medium_model = {
    "BartConfig.vocab_size": 4600,
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
    "TrainingArguments.learning_rate": 2e-4,
}

AlphaZeroModel = {
    "AlphaZeroPolicyConfig.num_residual_blocks" : 19,
    "AlphaZeroPolicyConfig.num_in_channels" : 5,
    "AlphaZeroPolicyConfig.num_out_channels" : 256,
    "AlphaZeroPolicyConfig.kernel_size" : 3,
    "AlphaZeroPolicyConfig.stride" : 1,
    "AlphaZeroPolicyConfig.board_size" : (19, 19),   
    "TrainingArguments.max_steps" : 5000,
    "TrainingArguments.per_device_train_batch_size": 256 ,
    "TrainingArguments.per_device_eval_batch_size": 256 ,
    "TrainingArguments.warmup_steps": 10,
    "TrainingArguments.weight_decay": 0.01,
    "TrainingArguments.logging_steps": 1,
    "TrainingArguments.evaluation_strategy": "steps",
    "TrainingArguments.eval_steps": 2,
    "TrainingArguments.learning_rate": 0.0002,
}

 