from typing import Type

# import evaluate

from data_processing.auxiliary_code_for_data_processing.pgn.chess_data_generator import ChessDataProvider
from jobs.core import Job

from metric_logging import source_files_register

source_files_register.register(__file__)


class CreatePGNDataset(Job):
    def __init__(
        self,
        chess_database_cls: Type[ChessDataProvider],
    ):
        self.chess_database = chess_database_cls()

    def execute(self) -> None:
        self.chess_database.create_data()
