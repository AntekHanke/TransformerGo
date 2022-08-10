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
        training_args=None,
        pgn_file=None,
        n_data=None,
        save_model_path=None
    ):
        self.model_config = model_config
        self.training_args = training_args

        self.model = BartForConditionalGeneration(self.model_config)
        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=ChessMovesDataGenerator(pgn_file=pgn_file, p_sample=0.5, n_data=n_data, mode="train"),
            eval_dataset=ChessMovesDataGenerator(pgn_file=pgn_file, p_sample=0.5, n_data=n_data//5, mode="eval"),
        )

        self.trainer.add_callback(NeptunePytorchCallback)

        assert save_model_path is not None

        self.save_model_path = save_model_path

    def execute(self):
        self.trainer.train()
        print(f'Saving model')
        self.trainer.save_model(self.save_model_path)
