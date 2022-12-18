from typing import Type, Union, List
from transformers.integrations import NeptuneCallback
from data_processing.pandas_data_provider import IterableDataLoader

from jobs.core import Job
from transformers import (
    Trainer,
    BartForConditionalGeneration,
    BartConfig,
    TrainingArguments,
)

from metric_logging import log_param, source_files_register, pytorch_callback_loggers
from utils.global_params_handler import GlobalParamsHandler

source_files_register.register(__file__)


class TrainModel(Job):
    def __init__(
        self,
        iterable_dataset_class: Type[IterableDataLoader],
        path_to_training_data: Union[str, List[str]],
        path_to_eval_data: Union[str, List[str]],
        take_random_half_of_training_data: bool = False,
        take_random_half_of_eval_data: bool = False,
        model_config_cls: Type[BartConfig] = None,
        training_args_cls: Type[TrainingArguments] = None,
        output_dir: str = None,
    ) -> None:

        x = GlobalParamsHandler()

        if GlobalParamsHandler().get_out_dir() is not None:
            output_dir = GlobalParamsHandler().get_out_dir()

        self.path_to_training_data = path_to_training_data
        self.path_to_eval_data = path_to_eval_data
        self.take_random_half_of_training_data = take_random_half_of_training_data
        self.take_random_half_of_eval_data = take_random_half_of_eval_data

        log_param("output_dir", output_dir)
        # TODO: How to fix typing and class initialization
        self.training_args = training_args_cls(output_dir=output_dir + "/out")
        if GlobalParamsHandler().learning_rate is not None:
            self.training_args.learning_rate = GlobalParamsHandler().learning_rate

        self.iterable_subgoal_dataLoader_train = iterable_dataset_class(
            data_path=self.path_to_training_data,
            take_random_half_of_data=self.take_random_half_of_training_data,
        )

        self.iterable_subgoal_dataLoader_eval = iterable_dataset_class(
            data_path=self.path_to_eval_data,
            take_random_half_of_data=self.take_random_half_of_eval_data,
        )

        self.model_config = model_config_cls()
        self.model = BartForConditionalGeneration(self.model_config)

        log_param("real learning rate", self.training_args.learning_rate)

        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.iterable_subgoal_dataLoader_train,
            eval_dataset=self.iterable_subgoal_dataLoader_eval,
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
