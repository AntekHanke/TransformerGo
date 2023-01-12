from typing import Type, Union, List, Optional
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
        path_to_training_data: Optional[str] = None,
        path_to_eval_data: Optional[str] = None,
        files_batch_size: int = 10,
        p_sample: float = 1.0,
        take_random_half_of_eval_data: bool = False,
        model_config_cls: Type[BartConfig] = None,
        training_args_cls: Type[TrainingArguments] = None,
        output_dir: str = None,
    ) -> None:

        global_params_handler: GlobalParamsHandler = GlobalParamsHandler()
        data_paths_from_gloabl_params_handler = global_params_handler.get_data_path()
        output_dir_from_global_params_handler = global_params_handler.get_out_dir()

        self.path_to_training_data = path_to_training_data
        self.path_to_eval_data = path_to_eval_data
        if data_paths_from_gloabl_params_handler is not None:
            self.path_to_training_data, self.path_to_eval_data = data_paths_from_gloabl_params_handler

        self.output_dir = output_dir
        if output_dir_from_global_params_handler is not None:
            self.output_dir = output_dir_from_global_params_handler

        self.take_random_half_of_training_data = p_sample
        self.take_random_half_of_eval_data = take_random_half_of_eval_data

        # TODO: How to fix typing and class initialization
        self.training_args = training_args_cls(output_dir=self.output_dir + "/out")
        if global_params_handler.learning_rate is not None:
            self.training_args.learning_rate = global_params_handler.learning_rate

        self.iterable_subgoal_dataLoader_train = iterable_dataset_class(
            data_path=self.path_to_training_data,
            files_batch_size=files_batch_size,
            p_sample=p_sample,
            cycle=True,
        )

        self.iterable_subgoal_dataLoader_eval = iterable_dataset_class(
            data_path=self.path_to_eval_data,
            files_batch_size=1,
            p_sample=1.0,
            cycle=False,
        )

        self.model_config = model_config_cls()
        self.model = BartForConditionalGeneration(self.model_config)

        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.iterable_subgoal_dataLoader_train,
            eval_dataset=self.iterable_subgoal_dataLoader_eval,
        )

        for callback_logger in pytorch_callback_loggers:
            self.trainer.add_callback(callback_logger.get_pytorch_callback())
        self.trainer.pop_callback(NeptuneCallback)
        self.save_model_path = self.output_dir + "/final_model"

        log_param("max learning rate", self.training_args.learning_rate)
        log_param("output_dir", self.output_dir)
        log_param("path_to_training_data", self.path_to_training_data)
        log_param("path_to_eval_data", self.path_to_eval_data)
        log_param("Save model path", self.save_model_path)

    def execute(self) -> None:
        self.trainer.train()
        print(f"Saving model at {self.save_model_path}")
        self.trainer.save_model(self.save_model_path)
