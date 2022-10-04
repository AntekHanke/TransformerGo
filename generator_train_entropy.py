
from transformers import (
    TrainingArguments,
    BartConfig,
)

from configs.global_config import ENTROPY_HOME
from metric_logging import log_param, register_logger, log_object, source_files_register
from data_processing.chess_data_generator import PolicyDataGenerator, ResultFilter, ChessSubgoalDataGenerator, NoFilter
from jobs.train_model import TrainModel
from mrunner_utils.mrunner_client import NeptuneLogger

source_files_register.register(__file__)


def train_generator_entropy(learning_rate, k, n_datapoints):

    # n_datapoints = 2*10 ** 7
    p_sample = 0.3

    print(f"learning_rate: {learning_rate}")

    LOG_DIR = f"{ENTROPY_HOME}/chess_models/generator_k={k}-lr_{learning_rate}/"

    log_param("learning_rate", learning_rate)
    log_param("log_dir", LOG_DIR)
    log_param("n_datapoints", n_datapoints)
    log_param("p_samples", p_sample)
    log_param("k", k)


    eagle_config = BartConfig(
        vocab_size=512,
        max_position_embeddings=128,
        encoder_layers=12,
        decoder_layers=12,
        encoder_attention_heads=8,
        decoder_attention_heads=8,
        decoder_ffn_dim=1024,
        encoder_ffn_dim=1024,
        d_model=512,
        dropout=0.
    )

    eagle_training = TrainingArguments(
        output_dir=LOG_DIR + "/out",  # output directory
        num_train_epochs=1,  # total number of training epochs
        per_device_train_batch_size=128,  # batch size per device during training
        per_device_eval_batch_size=128,  # batch size for evaluation
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir=LOG_DIR + "/results",  # directory for storing logs
        logging_steps=50,
        evaluation_strategy="steps",
        eval_steps=200,
        learning_rate=learning_rate,
    )

    chess_filter = ResultFilter('winner')
    # chess_filter = NoFilter()

    dataset = ChessSubgoalDataGenerator(
        k=k,
        pgn_file=f"{ENTROPY_HOME}/subgoal_chess/database.pgn",
        chess_filter=chess_filter,
        p_sample=p_sample,
        n_data=n_datapoints,
        log_samples_limit=50,
        p_log_sample=0.05,
    )

    dataset.create_data()

    log_param("save_model_path", LOG_DIR + "/generator_model")

    TrainModel(
        eagle_config,
        eagle_training,
        chess_database=dataset,
        save_model_path=LOG_DIR + "/generator_model"
    ).execute()
