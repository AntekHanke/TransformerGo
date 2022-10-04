
from transformers import (
    TrainingArguments,
    BartConfig,
)

from metric_logging import log_param, source_files_register
from data_processing.chess_data_generator import ResultFilter, ChessSubgoalDataGenerator
from jobs.train_model import TrainModel

source_files_register.register(__file__)


def train_generator_eagle(learning_rate, k, n_datapoints, p_sample):


    print(f"learning_rate: {learning_rate}")

    LOG_DIR = f"/home/plgrid/plgtodrzygozdz/chess_models/large_generator_k={k}-lr_{learning_rate}/"


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
        encoder_attention_heads=16,
        decoder_attention_heads=16,
        decoder_ffn_dim=2048,
        encoder_ffn_dim=2048,
        d_model=1024,
        dropout=0.05
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
        pgn_file="/home/plgrid/plgtodrzygozdz/subgoal_chess/database.pgn",
        chess_filter=chess_filter,
        p_sample=p_sample,
        n_data=n_datapoints,
        log_samples_limit=50,
        p_log_sample=0.05,
    )

    dataset.create_data()

    log_param("save_model_path", LOG_DIR + "/eagle_generator_model")

    TrainModel(
        eagle_config,
        eagle_training,
        chess_database=dataset,
        save_model_path=LOG_DIR + "/eagle_generator_model",
    ).execute()
