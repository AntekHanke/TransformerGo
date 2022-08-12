from config.global_config import source_files_register
from data_processing.chess_data_generator import PolicyDataGenerator
from jobs.core import Job
from transformers import (
    TrainingArguments,
    Trainer,
    BartForConditionalGeneration,
    BartConfig,
)

from mrunner_utils.neptune_logger import NeptunePytorchCallback

source_files_register.register(__file__)


class TrainModel(Job):
    def __init__(
        self,
        model_config=None,
        training_args=None,
        chess_database=None,
        save_model_path=None,
        neptune_logger=None
    ):
        self.model_config = model_config
        self.training_args = training_args

        self.model = BartForConditionalGeneration(self.model_config)
        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=chess_database.get_train_set_generator(),
            eval_dataset=chess_database.get_eval_set_generator(),
        )

        self.trainer.add_callback(neptune_logger.get_pytorch_callback())

        assert save_model_path is not None

        self.save_model_path = save_model_path

    def execute(self):
        self.trainer.train()
        print(f'Saving model')
        self.trainer.save_model(self.save_model_path)
