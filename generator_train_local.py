from pytorch_lightning import Trainer
from transformers import (
    TrainingArguments,
    Trainer,
    BartForConditionalGeneration,
    BartConfig,
)

import metric_logging
from data_processing.chess_data_generator import PolicyDataGenerator, ResultFilter, ChessSubgoalDataGenerator, NoFilter
from jobs.train_model import TrainModel
from mrunner_utils.mrunner_client import NeptuneLogger

metric_logging.source_files_register.register(__file__)
k = 2

LOCAL_LOG_DIR = f"/home/tomek/Research/subgoal_chess_data/generator_k_{k}/"
ENTROPY_LOG_DIR = "/home/todrzygozdz/subgoal_search_storage/"

EAGLE_PGN = "/home/plgrid/plgtodrzygozdz/subgoal_chess/database.pgn"
LOCAL_PGN = "/home/tomek/Research/subgoal_chess_data/database.pgn"


LOG_DIR = LOCAL_LOG_DIR

neptune_logger = NeptuneLogger(name=f"generator_k_1 train fast local", tags=["local", "generator_k_1"])
metric_logging.register_logger(neptune_logger)

fast_iter_config = BartConfig(
    vocab_size=512,
    max_position_embeddings=128,
    encoder_layers=8,
    decoder_layers=8,
    encoder_attention_heads=8,
    decoder_attention_heads=8,
    decoder_ffn_dim=512,
    encoder_ffn_dim=512,
    d_model=256,
    dropout=0.0,
)

fast_iter_training = TrainingArguments(
    output_dir=LOG_DIR + "/out",  # output directory
    num_train_epochs=1,  # total number of training epochs
    per_device_train_batch_size=64,  # batch size per device during training
    per_device_eval_batch_size=64,  # batch size for evaluation
    warmup_steps=400,  # number of warmup steps for learning rate scheduler
    weight_decay=0.01,  # strength of weight decay
    logging_dir=LOG_DIR + "/results",  # directory for storing logs
    logging_steps=5,
    evaluation_strategy="steps",
    eval_steps=10,
    learning_rate=3e-4,
)


chess_filter = ResultFilter("winner")
# chess_filter = NoFilter()

dataset = ChessSubgoalDataGenerator(
    k=k,
    pgn_file=LOCAL_PGN,
    chess_filter=chess_filter,
    p_sample=0.5,
    n_data=2*10**3,
    log_samples_limit=25,
    p_log_sample=0.05,
)

dataset.create_data()

TrainModel(
    fast_iter_config,
    fast_iter_training,
    chess_database=dataset,
    save_model_path=LOG_DIR + "/generator_model",
).execute()
