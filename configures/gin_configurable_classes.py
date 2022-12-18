import gin
from transformers import Trainer, TrainingArguments, BartConfig, BertConfig

from data_processing.chess_data_generator import (
    NoFilter,
    ResultFilter,
    ChessCLLPGamesDataGenerator,
    ChessSubgoalGamesDataGenerator,
    PolicyGamesDataGenerator,
)
from data_processing.mcts_data_generator import SubgoalMCGamesDataGenerator
from data_processing.pandas_data_provider import (
    IterableSubgoalDataLoader, IterablePolicyDataLoader
)
from jobs.create_pgn_dataset import CreatePGNDataset
from jobs.evaluate_cllp import EvaluateCLLP
from jobs.job_leela_dataset import LeelaCCLPDataProcessing, LeelaParallelDatasetGenerator
from jobs.train_bert_for_sequence_model import TrainBertForSequenceModel
from jobs.train_model import TrainModel


def configure_class(cls, module=None) -> None:
    gin.external_configurable(cls, module=module)


def configure_classes(classes, module=None) -> None:
    for cls in classes:
        configure_class(cls, module)


# configure_classes([GlobalParamsHandler], "params")
configure_classes(
    [
        TrainModel,
        CreatePGNDataset,
        LeelaCCLPDataProcessing,
        EvaluateCLLP,
        TrainBertForSequenceModel,
        LeelaParallelDatasetGenerator,
    ],
    "jobs",
)
configure_classes([Trainer, TrainingArguments, BartConfig, BertConfig], "transformers")
configure_classes([NoFilter, ResultFilter], "filters")
configure_classes(
    [
        PolicyGamesDataGenerator,
        ChessSubgoalGamesDataGenerator,
        ChessCLLPGamesDataGenerator,
        SubgoalMCGamesDataGenerator,
        IterableSubgoalDataLoader,
        IterablePolicyDataLoader
    ],
    "data",
)
