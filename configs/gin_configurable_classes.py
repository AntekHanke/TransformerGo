import gin
from transformers import Trainer, TrainingArguments, BartConfig

from data_processing.chess_data_generator import (
    NoFilter,
    ResultFilter,
    ChessCLLPDataGenerator,
    ChessSubgoalDataGenerator,
    PolicyDataGenerator,
)
from jobs.any_job import AnyJob
from jobs.train_model import TrainModel


def configure_class(cls, module=None):
    gin.external_configurable(cls, module=module)


def configure_classes(classes, module=None):
    for cls in classes:
        configure_class(cls, module)


configure_classes([AnyJob, TrainModel], "jobs")
configure_classes([Trainer, TrainingArguments, BartConfig], "transformers")
configure_classes([NoFilter, ResultFilter], "filters")
configure_classes([PolicyDataGenerator, ChessSubgoalDataGenerator, ChessCLLPDataGenerator], "data")
