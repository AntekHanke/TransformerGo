import gin
from transformers import Trainer, TrainingArguments, BartConfig, BertConfig

from data_processing.archive.pgn.mcts_data_generator import SubgoalMCGamesDataGenerator
from data_processing.archive.pgn.prepare_and_save_data import PandasPolicyPrepareAndSaveData, CLLPPrepareAndSaveData
from data_processing.chess_data_generator import (
    NoFilter,
    ResultFilter,
    ELOFilter,
    ChessCLLPGamesDataGenerator,
    ChessSubgoalGamesDataGenerator,
    PolicyGamesDataGenerator,
)
from data_processing.pandas_iterable_data_provider import (
    PandasIterableSubgoalDataProvider,
    PandasIterablePolicyDataProvider,
    PandasIterablePolicyWithHistoryDataProvider,
    PandasIterablePolicyOnlyHistoryDataProvider,
    PandasBertForSequenceDataProvider,
    PandasIterableSubgoalToPolicyDataProvider,
    PandasIterableCLLPDataProvider,
)
from data_processing.pandas_static_dataset_provider import (
    PandasStaticDataProvider,
    PandasStaticSubgoalDataProvider,
    PandasStaticPolicyDataProvider,
    PandasStaticPolicyWithHistoryDataProvider,
    PandasStaticPolicyOnlyHistoryDataProvider,
    PandasStaticSubgoalToPolicyDataProvider,
    PandasStaticCLLPDataProvider,
)
from jobs.chess_retokenization import RetokenizationJob
from jobs.create_pgn_dataset import CreatePGNDataset
from jobs.debug_job import DebugJob
from jobs.job_leela_dataset import LeelaCCLPDataProcessing, LeelaParallelDatasetGenerator, LeelaPrepareAndSaveData
from jobs.train_bert_for_sequence_model import TrainBertForSequenceModel
from jobs.train_model import TrainModelFromScratch


def configure_class(cls, module=None) -> None:
    gin.external_configurable(cls, module=module)


def configure_classes(classes, module=None) -> None:
    for cls in classes:
        configure_class(cls, module)


# configure_classes([GlobalParamsHandler], "params")
configure_classes(
    [
        DebugJob,
        TrainModelFromScratch,
        CreatePGNDataset,
        LeelaCCLPDataProcessing,
        TrainBertForSequenceModel,
        LeelaParallelDatasetGenerator,
        LeelaPrepareAndSaveData,
        RetokenizationJob,
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
        PandasIterablePolicyWithHistoryDataProvider,
        PandasIterablePolicyOnlyHistoryDataProvider,
        PandasIterableSubgoalToPolicyDataProvider,
        PandasIterableCLLPDataProvider,
        PandasStaticDataProvider,
        PandasStaticSubgoalDataProvider,
        PandasStaticPolicyDataProvider,
        PandasStaticPolicyWithHistoryDataProvider,
        PandasStaticPolicyOnlyHistoryDataProvider,
        PandasStaticSubgoalToPolicyDataProvider,
        PandasStaticCLLPDataProvider,
        PandasPolicyPrepareAndSaveData,
        PandasBertForSequenceDataProvider,
        CLLPPrepareAndSaveData,
    ],
    "data",
)
