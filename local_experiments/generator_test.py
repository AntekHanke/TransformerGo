import chess

from data_structures.data_structures import ImmutableBoard
from jobs.evaluate_generator import EvaluateGenerator
from metric_logging import source_files_register, register_logger
from mrunner_utils.mrunner_client import NeptuneLogger
from subgoal_generator.subgoal_generator import BasicChessSubgoalGenerator

generator = BasicChessSubgoalGenerator("/generator_k_2/generator_model")

source_files_register.register(__file__)

neptune_logger = NeptuneLogger(
    name=f"evaluate generator", tags=["generator", "eval"], params={}, properties={}
, git_info=None)

register_logger(neptune_logger)

job = EvaluateGenerator(2, 3, generator, "/database.pgn", 15, 100)
job.execute()