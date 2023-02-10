import gin
from transformers import Trainer, TrainingArguments, BartConfig, BertConfig

from data_processing.chess_data_generator import (
    NoFilter,
    ResultFilter,
    ELOFilter,
    ChessCLLPGamesDataGenerator,
    ChessSubgoalGamesDataGenerator,
    PolicyGamesDataGenerator,
)
from data_processing.archive.pgn.mcts_data_generator import SubgoalMCGamesDataGenerator
from data_processing.pandas_iterable_data_provider import (
    PandasIterableSubgoalDataProvider, PandasIterablePolicyDataProvider, PandasBertForSequenceDataProvider,
    PandasIterableSubgoalToPolicyDataProvider, PandasIterableCLLPDataProvider
)
from data_processing.pandas_static_dataset_provider import PandasStaticDataProvider, PandasStaticSubgoalDataProvider, \
    PandasStaticPolicyDataProvider, PandasStaticSubgoalToPolicyDataProvider, PandasStaticCLLPDataProvider
from data_processing.archive.pgn.prepare_and_save_data import PandasPolicyPrepareAndSaveData, CLLPPrepareAndSaveData
from jobs.create_pgn_dataset import CreatePGNDataset
from jobs.debug_job import DebugJob
from jobs.job_leela_dataset import LeelaCCLPDataProcessing, LeelaParallelDatasetGenerator, LeelaPrepareAndSaveData
from jobs.train_bert_for_sequence_model import TrainBertForSequenceModel
from jobs.train_model import TrainModel
from jobs.chess_retokenization import RetokenizationJob

def configure_class(cls, module=None) -> None:
    gin.external_configurable(cls, module=module)


def configure_classes(classes, module=None) -> None:
    for cls in classes:
        configure_class(cls, module)


# configure_classes([GlobalParamsHandler], "params")
configure_classes(
    [
        DebugJob,
        TrainModel,
        CreatePGNDataset,
        LeelaCCLPDataProcessing,
        TrainBertForSequenceModel,
        LeelaParallelDatasetGenerator,
        LeelaPrepareAndSaveData,
        RetokenizationJob
    ],
    "jobs",
)
configure_classes([Trainer, TrainingArguments, BartConfig, BertConfig], "transformers")
configure_classes([NoFilter, ResultFilter, ELOFilter], "filters")
configure_classes(
    [
        PolicyGamesDataGenerator,
        ChessSubgoalGamesDataGenerator,
        ChessCLLPGamesDataGenerator,
        SubgoalMCGamesDataGenerator,
        PandasIterableSubgoalDataProvider,
        PandasIterablePolicyDataProvider,
        PandasIterableSubgoalToPolicyDataProvider,
        PandasIterableCLLPDataProvider,
        PandasStaticSubgoalDataProvider,
        PandasStaticPolicyDataProvider,
        PandasStaticSubgoalToPolicyDataProvider,
        PandasStaticCLLPDataProvider,
        PandasPolicyPrepareAndSaveData,
        PandasBertForSequenceDataProvider,

        CLLPPrepareAndSaveData
    ],
    "data",
)
