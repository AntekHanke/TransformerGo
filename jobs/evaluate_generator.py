import random

from data_processing.chess_data_generator import ChessDataGenerator, NoFilter, ChessSubgoalDataGenerator
from data_processing.chess_tokenizer import ChessTokenizer
from data_processing.data_utils import immutable_boards_to_img
from jobs.core import Job
from transformers import (
    Trainer,
    BartForConditionalGeneration,
    BartConfig,
    TrainingArguments,
)

from metric_logging import log_param, source_files_register, log_object
from mrunner_utils.neptune_logger import NeptuneLogger
from subgoal_generator.subgoal_generator import ChessSubgoalGenerator

source_files_register.register(__file__)


class EvaluateGenerator(Job):
    def __init__(
        self,
        k: int,
        n_subgoals: int,
        subgoal_generator: ChessSubgoalGenerator,
        pgn_file: str = None,
        n_eval_datapoints: int = 1000,
        n_log_samples: int = 100,
    ):

        self.k = k
        self.n_subgoals = n_subgoals
        self.subgoal_generator = subgoal_generator
        self.chess_database = ChessSubgoalDataGenerator(self.k,
            pgn_file, NoFilter(), p_sample=0.5, n_data=100 * n_eval_datapoints, only_eval=True
        )
        self.n_eval_datapoints = n_eval_datapoints
        self.n_log_samples = n_log_samples

    def execute(self):
        self.chess_database.create_data()
        eval_data = self.chess_database.get_eval_set_generator()
        eval_data_len = len(eval_data)
        idx_to_eval = random.choices(list(range(eval_data_len)), k=self.n_eval_datapoints)
        for idx in idx_to_eval:
            input_board = ChessTokenizer.decode_board(eval_data[idx]["input_ids"])
            data_target = ChessTokenizer.decode_board(eval_data[idx]["labels"])
            subgoals = self.subgoal_generator.generate_subgoals(input_board, self.n_subgoals)
            fig = immutable_boards_to_img(
                [input_board, data_target, *subgoals],
                ["input", "data"] + ["subgoal_{}".format(i) for i in range(self.n_log_samples)],
            )
            log_object("Subgoals", fig)
