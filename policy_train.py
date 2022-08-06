from pytorch_lightning import Trainer
from transformers import (
    TrainingArguments,
    Trainer,
    BartForConditionalGeneration,
    BartConfig,
)

from config.global_config import source_files_register
from data_processing.chess_data_generator import ChessMovesDataGenerator
from jobs.train_model import TrainModel

source_files_register.register(__file__)

model =

small_config = BartConfig(
    vocab_size=128,
    max_position_embeddings=128,
    encoder_ffn_dim=512,
    decoder_ffn_dim=512,
)
training_args = TrainingArguments(
    output_dir="./results",  # output directory
    num_train_epochs=3,  # total number of training epochs
    per_device_train_batch_size=2,  # batch size per device during training
    per_device_eval_batch_size=2,  # batch size for evaluation
    warmup_steps=50,  # number of warmup steps for learning rate scheduler
    weight_decay=0.01,  # strength of weight decay
    logging_dir="out",  # directory for storing logs
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=10,
)
trainer = Trainer(
    model=self.model,
    args=self.training_args,
    train_dataset=ChessMovesDataGenerator(
        "/home/tomek/Research/subgoal_chess_data/database.pgn"
    ),
    eval_dataset=ChessMovesDataGenerator(
        "/home/tomek/Research/subgoal_chess_data/database.pgn"
    ),
)

x = TrainModel(small_config, training_args, None)
x.execute()
