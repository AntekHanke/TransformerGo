from config.global_config import source_files_register
from data_processing.chess_data_generator import ChessMovesDataGenerator
from jobs.core import Job
from transformers import (
    TrainingArguments,
    Trainer,
    BartForConditionalGeneration,
    BartConfig,
)

from mrunner_utils.neptune_callback import NeptunePytorchCallback

source_files_register.register(__file__)


class TrainModel(Job):
    def __init__(
        self,
        model_config=None,
        trainer=None,
        training_args=None,
        data_generation_class=None,
    ):
        self.model_config = model_config
        self.training_args = training_args
        self.data_generator = data_generation_class

        self.model = BartForConditionalGeneration(self.model_config)
        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=ChessMovesDataGenerator(
                "/home/tomek/Research/subgoal_chess_data/database.pgn"
            ),
            eval_dataset=ChessMovesDataGenerator(
                "/home/tomek/Research/subgoal_chess_data/database.pgn"
            ),
        )
        self.trainer.add_callback(NeptunePytorchCallback)

    def execute(self):
        self.trainer.train()
