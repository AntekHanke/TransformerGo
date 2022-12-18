from mrunner.helpers.specification_helper import create_experiments_helper


base_config = {
    "run.job_class": "@jobs.TrainBertForSequenceModel",
    "TrainBertForSequenceModel.chess_database_cls": "@data.PandasBertForSequenceDataProvider",
    "TrainBertForSequenceModel.model_config_cls": "@transformers.BertConfig",
    "TrainBertForSequenceModel.training_args_cls": "@transformers.TrainingArguments",
    "PandasBertForSequenceDataProvider.data_path": "/leela_generator_data/full_dataset_k=2.pkl",
    "PandasBertForSequenceDataProvider.eval_datapoints": 1000,
    "GlobalParamsHandler.out_dir": "/leela_models/v1/bert_for_sequence/small_model",
    # "GlobalParamsHandler.data_location": "/leela_bert_data/prom_full_dataset_k=3.pkl",
    # # "GlobalParamsHandler.data_type": "cllp",
    # "GlobalParamsHandler.path_format": ["learning_rate"],
    "BertConfig.num_labels": 1,
    "BertConfig.vocab_size": 512,
    "BertConfig.hidden_size": 256,
    "BertConfig.num_hidden_layers": 2,
    "BertConfig.num_attention_heads": 2,
    "BertConfig.intermediate_size": 128,
    "BertConfig.hidden_act": "gelu",
    "BertConfig.hidden_dropout_prob": 0.1,
    "BertConfig.attention_probs_dropout_prob": 0.1,
    "BertConfig.max_position_embeddings": 256,
    "BertConfig.layer_norm_eps": 1e-12,
    "BertConfig.position_embedding_type": "absolute",
    "BertConfig.classifier_dropout": 0.1,

    "TrainingArguments.num_train_epochs": 1,
    "TrainingArguments.per_device_train_batch_size": 1024,
    "TrainingArguments.per_device_eval_batch_size": 1024,
    "TrainingArguments.warmup_steps": 100,
    "TrainingArguments.weight_decay": 0.01,
    "TrainingArguments.logging_steps": 10,
    "TrainingArguments.evaluation_strategy": "steps",
    "TrainingArguments.eval_steps": 5,
    "TrainingArguments.learning_rate": 0.0002,
    "use_neptune": True,
}

params_grid = {
    "idx": [0],
    # "GlobalParamsHandler.k": [3]
}

experiments_list = create_experiments_helper(
    experiment_name=f"ultra-small-leela-bert-train",
    project_name="pmtest/subgoal-chess",
    base_config=base_config,
    params_grid=params_grid,
    script="python3 -m runner --mrunner",
    exclude=["data", ".pytest_cache", "out", ".git"],
    python_path="",
    tags=["leela", "train", "ultra-small"],
    with_neptune=True,
    env={},
)
