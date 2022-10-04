import gin
from transformers import Trainer, TrainingArguments, BartConfig

from data_processing.chess_data_generator import (
    NoFilter,
    ResultFilter,
    ChessCLLPGamesDataGenerator,
    ChessSubgoalGamesDataGenerator,
    PolicyGamesDataGenerator,
)
from jobs.any_job import AnyJob
from jobs.train_model import TrainModel


def configure_class(cls, module=None) -> None:
    gin.external_configurable(cls, module=module)


def configure_classes(classes, module=None) -> None:
    for cls in classes:
        configure_class(cls, module)


configure_classes([AnyJob, TrainModel], "jobs")
configure_classes([Trainer, TrainingArguments, BartConfig], "transformers")
configure_classes([NoFilter, ResultFilter], "filters")
configure_classes([PolicyGamesDataGenerator, ChessSubgoalGamesDataGenerator, ChessCLLPGamesDataGenerator], "data")
