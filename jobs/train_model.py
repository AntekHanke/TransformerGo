from typing import Type, Union

# import evaluate

from transformers.integrations import NeptuneCallback

from data_processing.chess_data_generator import ChessGamesDataGenerator, ChessDataProvider
from jobs.core import Job
from transformers import (
    Trainer,
    BartForConditionalGeneration,
    BartConfig,
    TrainingArguments,
)

from metric_logging import log_param, source_files_register, pytorch_callback_loggers
from utils.global_params_handler import GlobalParamsHandler


class TrainModel(Job):
    def __init__(
        self,
        chess_database_cls: Type[ChessDataProvider],
        model_config_cls: Type[BartConfig] = None,
        training_args_cls: Type[TrainingArguments] = None,
        output_dir: str = None,
    ):

        if GlobalParamsHandler().get_out_dir() is not None:
            output_dir = GlobalParamsHandler().get_out_dir()

        log_param("output_dir", output_dir)

        chess_database = chess_database_cls()
        if isinstance(chess_database, ChessGamesDataGenerator):
            chess_database.create_data()

        self.model_config = model_config_cls()
        self.training_args = training_args_cls(output_dir=output_dir + "/out")

        if GlobalParamsHandler().learning_rate is not None:
            self.training_args.learning_rate = GlobalParamsHandler().learning_rate

        self.model = BartForConditionalGeneration(self.model_config)

        log_param("real learning rate", self.training_args.learning_rate)

        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=chess_database.get_train_set_generator(),
            eval_dataset=chess_database.get_eval_set_generator(),
            # compute_metrics=compute_metrics,
        )

        for callback_logger in pytorch_callback_loggers:
            self.trainer.add_callback(callback_logger.get_pytorch_callback())
        self.trainer.pop_callback(NeptuneCallback)
        self.save_model_path = output_dir + "/final_model"
        log_param("Save model path", self.save_model_path)

    def execute(self) -> None:
        self.trainer.train()
        print(f"Saving model at {self.save_model_path}")
        self.trainer.save_model(self.save_model_path)
