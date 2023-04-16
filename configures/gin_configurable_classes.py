import gin
from transformers import Trainer, TrainingArguments, BartConfig, BertConfig
from go_policy.policy_config import AlphaZeroPolicyConfig
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
    PandasIterablePolicyDataProviderGo,
    PandasIterablePolicyWithHistoryDataProvider,
    PandasBertForSequenceDataProvider,
    PandasIterableSubgoalToPolicyDataProvider,
    PandasIterableCLLPDataProvider,
    PandasIterableSubgoalAllDistancesDataProvider,
    PandasIterablePolicyDataProviderGo
)
from data_processing.pandas_static_dataset_provider import (
    PandasStaticDataProvider,
    PandasStaticSubgoalDataProvider,
    PandasStaticPolicyDataProvider,
    PandasStaticPolicyDataProviderGo,
    PandasStaticPolicyWithHistoryDataProvider,
    PandasStaticSubgoalToPolicyDataProvider,
    PandasStaticCLLPDataProvider,
    PandasStaticSubgoalAllDistancesDataProvider,
    PandasStaticPolicyDataProviderGo
)
from jobs.chess_retokenization import RetokenizationJob
from jobs.create_pgn_dataset import CreatePGNDataset
from jobs.debug_job import DebugJob
from jobs.job_leela_dataset import (
    LeelaCCLPDataProcessing,
    LeelaParallelDatasetGenerator,
    LeelaPrepareAndSaveData,
)
from jobs.job_leela_dataset import LeelaCCLPDataProcessing, LeelaParallelDatasetGenerator, LeelaPrepareAndSaveData
from jobs.local_jobs_antek.go_data_generator_tokenized_policy import GoTokenizedPolicyGeneratorAlwaysBlack
from jobs.go_convolution_data_generation import GoConvolutionDataGeneration
from jobs.go_train_convolutions import GoTrainConvolution 
from jobs.train_bert_for_sequence_model import TrainBertForSequenceModel
from jobs.go_train_bert_for_sequence_model import GoTrainBertForSequenceModel
from jobs.train_model import TrainModelFromScratch, ResumeTraining, TrainConvolutionFromScratch
# from jobs.run_mcts import RunMCTSJob
# from mcts.mcts import score_function, expand_function, mock_expand_function

from data_processing.go_data_generator import GoSimpleGamesDataGeneratorTokenizedAlwaysBlack, SimpleGamesDataGenerator


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
        GoTokenizedPolicyGeneratorAlwaysBlack,
        GoTrainBertForSequenceModel,
        # RunMCTSJob,
        GoConvolutionDataGeneration,
        GoTrainConvolution,
        TrainConvolutionFromScratch
    ],
    "jobs",
)
configure_class(AlphaZeroPolicyConfig, "AlphaZero")
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
        PandasIterablePolicyDataProviderGo,
        PandasIterablePolicyWithHistoryDataProvider,
        PandasIterableSubgoalToPolicyDataProvider,
        PandasIterableCLLPDataProvider,
        PandasStaticDataProvider,
        PandasStaticSubgoalDataProvider,
        PandasStaticPolicyDataProvider,
        PandasStaticPolicyDataProviderGo,
        PandasStaticPolicyWithHistoryDataProvider,
        PandasStaticSubgoalToPolicyDataProvider,
        PandasStaticCLLPDataProvider,
        PandasPolicyPrepareAndSaveData,
        PandasBertForSequenceDataProvider,
        CLLPPrepareAndSaveData,
        GoSimpleGamesDataGeneratorTokenizedAlwaysBlack,
        SimpleGamesDataGenerator,
        PandasIterablePolicyDataProviderGo,
        PandasStaticPolicyDataProviderGo
    ],
    "data",
)

# configure_objects([expand_function, mock_expand_function], "expand_functions")
# configure_objects([score_function], "score_functions")
