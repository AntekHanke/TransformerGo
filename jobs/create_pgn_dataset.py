from typing import Type, Union

# import evaluate
from transformers.integrations import NeptuneCallback

from data_processing.chess_data_generator import ChessGamesDataGenerator, ChessDataProvider
from jobs.core import Job
from transformers import (
    Trainer,
    BartForConditionalGeneration,
    BartConfig,
    TrainingArguments,
)

from metric_logging import log_param, source_files_register, pytorch_callback_loggers

source_files_register.register(__file__)


class CreatePGNDataset(Job):
    def __init__(
        self,
        chess_database_cls: Type[ChessDataProvider],
    ):

        self.chess_database = chess_database_cls()



    def execute(self) -> None:
        self.chess_database.create_data()
