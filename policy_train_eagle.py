import gin
from pytorch_lightning import Trainer
from transformers import (
    TrainingArguments,
    Trainer,
    BartForConditionalGeneration,
    BartConfig,
)

import metric_logging
from configs.global_config import source_files_register
from data_processing.chess_data_generator import PolicyDataGenerator
from jobs.train_model import TrainModel
from mrunner_utils.neptune_logger import NeptunePytorchCallback, NeptuneLogger

source_files_register.register(__file__)

# LOG_DIR = ENTROPY_LOG_DIR

# learning_rate = 1e-5


def train_policy_eagle(learning_rate):

    print(f"learning_rate: {learning_rate}")

    LOG_DIR = f"/home/plgrid/plgtodrzygozdz/chess_models/policy-lr_{learning_rate}/"

    neptune_logger = NeptuneLogger(
        name=f"eagle_policy_train-{learning_rate} lr", tags=["eagle", "policy"]
    )
    metric_logging.register_logger(neptune_logger)

    metric_logging.log_object("learning_rate", f"{learning_rate}")

    eagle_config = BartConfig(
        vocab_size=512,
        max_position_embeddings=128,
        encoder_layers=10,
        decoder_layers=10,
        encoder_attention_heads=8,
        decoder_attention_heads=8,
        decoder_ffn_dim=1024,
        encoder_ffn_dim=1024,
        d_model=256,
        dropout=0.
    )

    eagle_training = TrainingArguments(
        output_dir=LOG_DIR + "/out",  # output directory
        num_train_epochs=1,  # total number of training epochs
        per_device_train_batch_size=2048,  # batch size per device during training
        per_device_eval_batch_size=2048,  # batch size for evaluation
        warmup_steps=1000,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir=LOG_DIR + "/results",  # directory for storing logs
        logging_steps=50,
        evaluation_strategy="steps",
        eval_steps=100,
        learning_rate=learning_rate,
    )

    dataset = PolicyDataGenerator(
        pgn_file="/home/plgrid/plgtodrzygozdz/subgoal_chess/database.pgn",
        p_sample=0.2,
        n_data=10**5,
        log_samples_limit=10,
    )

    TrainModel(
        fast_iter_config,
        fast_iter_training,
        chess_database=dataset,
        save_model_path=LOG_DIR + "/eagle_policy_model",
        neptune_logger=neptune_logger,
    ).execute()
