from pytorch_lightning import Trainer
from transformers import (
    TrainingArguments,
    Trainer,
    BartForConditionalGeneration,
    BartConfig,
)

from config.global_config import source_files_register
from data_processing.chess_data_generator import PolicyDataGenerator
from jobs.train_model import TrainModel

source_files_register.register(__file__)

LOCAL_LOG_DIR = "/home/tomek/Research/subgoal_chess_data"
ENTROPY_LOG_DIR = "/home/todrzygozdz/subgoal_search_storage/"


LOG_DIR = LOCAL_LOG_DIR


small_config = BartConfig(
    vocab_size=128,
    max_position_embeddings=128,
    encoder_ffn_dim=512,
    decoder_ffn_dim=512,
    decoder_layers=4,
    encoder_layers=4,
    encoder_attention_heads=4,
    decoder_attention_heads=4,
    d_model=256,
)

medium_config = BartConfig(
    vocab_size=128,
    max_position_embeddings=128,
    encoder_layers=10,
    encoder_ffn_dim=1024,
    decoder_layers=10,
    decoder_ffn_dim=1024,
    d_model=128,
)

training_args = TrainingArguments(
    output_dir=LOG_DIR + "/out",  # output directory
    num_train_epochs=5,  # total number of training epochs
    per_device_train_batch_size=128,  # batch size per device during training
    per_device_eval_batch_size=128,  # batch size for evaluation
    warmup_steps=500,  # number of warmup steps for learning rate scheduler
    weight_decay=0.01,  # strength of weight decay
    logging_dir=LOG_DIR + "/out",  # directory for storing logs
    logging_steps=1000,
    evaluation_strategy="steps",
    eval_steps=2000,
)


x = TrainModel(
    small_config,
    training_args,
    pgn_file="/home/tomek/Research/subgoal_chess_data/database.pgn",
    n_data=10**6,
    save_model_path="/home/tomek/Research/subgoal_chess_data/local_policy",
)

# x = TrainModel(
#     model_config=medium_config,
#     training_args=training_args,
#     pgn_file="/home/todrzygozdz/subgoal_chess_data/database.pgn",
#     n_data=500000,
#     save_model_path="/home/todrzygozdz/subgoal_chess_models/policy",
# )

x.execute()
