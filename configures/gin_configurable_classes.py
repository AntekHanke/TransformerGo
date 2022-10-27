import gin
from transformers import Trainer, TrainingArguments, BartConfig

from data_processing.chess_data_generator import (
    NoFilter,
    ResultFilter,
    ChessCLLPGamesDataGenerator,
    ChessSubgoalGamesDataGenerator,
    PolicyGamesDataGenerator,
)
from data_processing.mcts_data_generator import SubgoalMCGamesDataGenerator
from data_processing.pandas_data_provider import PandasSubgoalDataProvider, PandasCLLPDataGenerator
from jobs.any_job import AnyJob
from jobs.create_pgn_dataset import CreatePGNDataset

# from jobs.job_leela_dataset import LeelaDatasetGenerator
from jobs.job_leela_dataset import LeelaCCLPDataProcessing
from jobs.train_model import TrainModel


def configure_class(cls, module=None) -> None:
    gin.external_configurable(cls, module=module)


def configure_classes(classes, module=None) -> None:
    for cls in classes:
        configure_class(cls, module)


# configure_classes([GlobalParamsHandler], "params")
configure_classes([AnyJob, TrainModel, CreatePGNDataset, LeelaCCLPDataProcessing], "jobs")
configure_classes([Trainer, TrainingArguments, BartConfig], "transformers")
configure_classes([NoFilter, ResultFilter], "filters")
configure_classes(
    [
        PolicyGamesDataGenerator,
        ChessSubgoalGamesDataGenerator,
        ChessCLLPGamesDataGenerator,
        SubgoalMCGamesDataGenerator,
        PandasSubgoalDataProvider,
        PandasCLLPDataGenerator
    ],
    "data",
)
