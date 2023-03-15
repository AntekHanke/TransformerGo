import gin
from transformers import Trainer, TrainingArguments, BartConfig, BertConfig

from data_processing.archive.pgn.mcts_data_generator import SubgoalMCGamesDataGenerator
from data_processing.archive.pgn.prepare_and_save_data import (
    PandasPolicyPrepareAndSaveData,
    CLLPPrepareAndSaveData,
)
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
    PandasBertForSequenceDataProvider,
    PandasIterableSubgoalToPolicyDataProvider,
    PandasIterableCLLPDataProvider,
    PandasIterableSubgoalAllDistancesDataProvider,
)
from data_processing.pandas_static_dataset_provider import (
    PandasStaticDataProvider,
    PandasStaticSubgoalDataProvider,
    PandasStaticPolicyDataProvider,
    PandasStaticPolicyWithHistoryDataProvider,
    PandasStaticSubgoalToPolicyDataProvider,
    PandasStaticCLLPDataProvider,
    PandasStaticSubgoalAllDistancesDataProvider,
)
from jobs.chess_retokenization import RetokenizationJob
from jobs.create_pgn_dataset import CreatePGNDataset
from jobs.debug_job import DebugJob
from jobs.job_leela_dataset import (
    LeelaCCLPDataProcessing,
    LeelaParallelDatasetGenerator,
    LeelaPrepareAndSaveData,
)
from jobs.train_bert_for_sequence_model import TrainBertForSequenceModel
from jobs.train_model import TrainModelFromScratch, ResumeTraining
from jobs.run_mcts import RunMCTSJob
from mcts.mcts import score_function, mock_expand_function, StandardExpandFunction
from mcts.node_expansion import ChessStateExpander
from policy.chess_policy import LCZeroPolicy
from policy.cllp import CLLP
from subgoal_generator.subgoal_generator import BasicChessSubgoalGenerator
from value.chess_value import LCZeroValue


def configure_class(cls, module=None) -> None:
    gin.external_configurable(cls, module=module)


def configure_classes(classes, module=None) -> None:
    for cls in classes:
        configure_class(cls, module)


def configure_object(obj, module=None) -> None:
    gin.external_configurable(obj, module=module)


def configure_objects(objects, module=None) -> None:
    for obj in objects:
        configure_object(obj, module)


configure_classes(
    [
        DebugJob,
        TrainModelFromScratch,
        ResumeTraining,
        CreatePGNDataset,
        LeelaCCLPDataProcessing,
        TrainBertForSequenceModel,
        LeelaParallelDatasetGenerator,
        LeelaPrepareAndSaveData,
        RetokenizationJob,
        RunMCTSJob,
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
        PandasIterableSubgoalAllDistancesDataProvider,
        PandasStaticSubgoalAllDistancesDataProvider,
        PandasIterablePolicyDataProvider,
        PandasIterablePolicyWithHistoryDataProvider,
        PandasIterableSubgoalToPolicyDataProvider,
        PandasIterableCLLPDataProvider,
        PandasStaticDataProvider,
        PandasStaticSubgoalDataProvider,
        PandasStaticPolicyDataProvider,
        PandasStaticPolicyWithHistoryDataProvider,
        PandasStaticSubgoalToPolicyDataProvider,
        PandasStaticCLLPDataProvider,
        PandasPolicyPrepareAndSaveData,
        PandasBertForSequenceDataProvider,
        CLLPPrepareAndSaveData,
    ],
    "data",
)

configure_objects([StandardExpandFunction], "expand_functions")
configure_objects([score_function], "score_functions")
configure_class(ChessStateExpander, "chess_state_expander")
configure_classes([LCZeroPolicy, LCZeroValue, CLLP, BasicChessSubgoalGenerator], "neural_networks")
