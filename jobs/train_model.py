from typing import Type, Union

from data_processing.chess_data_generator import ChessGamesDataGenerator, ChessDataProvider
from jobs.core import Job
from transformers import (
    Trainer,
    BartForConditionalGeneration,
    BartConfig,
    TrainingArguments,
)

from metric_logging import log_param, source_files_register, pytorch_callback_loggers

source_files_register.register(__file__)


class TrainModel(Job):
    def __init__(
        self,
        chess_database_cls: Type[ChessDataProvider],
        model_config: Union[BartConfig, None] = None,
        training_args: Union[TrainingArguments] = None,
        save_model_path: str = None,
    ):
        if model_config is None:
            model_config = BartConfig()
        if training_args is None:
            training_args = TrainingArguments(output_dir=f"{save_model_path}/out")

        chess_database = chess_database_cls()
        if isinstance(chess_database, ChessGamesDataGenerator):
            chess_database.create_data()

        self.model_config = model_config
        self.training_args = training_args

        self.model = BartForConditionalGeneration(self.model_config)
        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=chess_database.get_train_set_generator(),
            eval_dataset=chess_database.get_eval_set_generator(),
        )

        for callback_logger in pytorch_callback_loggers:
            self.trainer.add_callback(callback_logger.get_pytorch_callback())
        assert save_model_path is not None
        self.save_model_path = save_model_path + "/final_model"
        log_param("Save model path", self.save_model_path)

    def execute(self) -> None:
        self.trainer.train()
        print(f"Saving model at {self.save_model_path}")
        self.trainer.save_model(self.save_model_path)
