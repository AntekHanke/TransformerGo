from typing import Type, Optional

import numpy as np
import torch
from transformers import (
    Trainer,
    BartForConditionalGeneration,
    BartConfig,
    TrainingArguments,
)
from transformers.integrations import NeptuneCallback

from data_processing.pandas_iterable_data_provider import PandasIterableDataProvider
from data_processing.pandas_static_dataset_provider import PandasStaticDataProvider
from jobs.core import Job
from metric_logging import log_param, source_files_register, pytorch_callback_loggers
from utils.global_params_handler import GlobalParamsHandler

source_files_register.register(__file__)


class TrainModel(Job):
    def __init__(
            self,
            train_data_provider: Type[PandasIterableDataProvider],
            eval_data_provider: Type[PandasStaticDataProvider],
            path_to_training_data: Optional[str] = None,
            path_to_eval_data: Optional[str] = None,
            files_batch_size: int = 10,
            eval_n_batches: int = 10,
            prob_take_sample: float = 1.0,
            model_config_cls: Type[BartConfig] = None,
            training_args_cls: Type[TrainingArguments] = None,
            out_dir: str = None,
    ) -> None:

        global_params_handler: GlobalParamsHandler = GlobalParamsHandler()
        data_paths_from_gloabl_params_handler = global_params_handler.get_data_path()
        output_dir_from_global_params_handler = global_params_handler.get_out_dir()

        self.path_to_training_data = path_to_training_data
        self.path_to_eval_data = path_to_eval_data
        if data_paths_from_gloabl_params_handler is not None:
            self.path_to_training_data, self.path_to_eval_data = data_paths_from_gloabl_params_handler

        self.out_dir = out_dir
        if output_dir_from_global_params_handler is not None:
            self.out_dir = output_dir_from_global_params_handler

        self.training_args = training_args_cls(output_dir=self.out_dir + "/out")
        if global_params_handler.learning_rate is not None:
            self.training_args.learning_rate = global_params_handler.learning_rate

        self.iterable_subgoal_dataLoader_train = train_data_provider(
            data_path=self.path_to_training_data,
            files_batch_size=files_batch_size,
            p_sample=prob_take_sample,
            cycle=True,
            name="train",
        )

        self.iterable_subgoal_dataLoader_eval = eval_data_provider(
            data_path=self.path_to_eval_data,
            files_batch_size=files_batch_size,
            p_sample=prob_take_sample,
            eval_num_samples=eval_n_batches * training_args_cls.per_device_eval_batch_size,
            name="eval",
        )

        self.model_config = model_config_cls()
        self.model = BartForConditionalGeneration(self.model_config)

        def preprocess_logits_for_metrics(logits, labels):
            pred_ids = torch.argmax(logits[0], dim=-1)
            return pred_ids, labels

        def compute_metrics(eval_preds):
            predictions, labels = eval_preds
            return {
                "accuracy": (predictions == labels).astype(np.float32).mean().item(),
                "perfect_sequence": (predictions == labels).all(axis=1).astype(np.float32).mean().item(),
            }

        print("Training arguments:")
        print(self.training_args)

        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.iterable_subgoal_dataLoader_train,
            eval_dataset=self.iterable_subgoal_dataLoader_eval,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            compute_metrics=compute_metrics,
        )

        for callback_logger in pytorch_callback_loggers:
            self.trainer.add_callback(callback_logger.get_pytorch_callback())
        self.trainer.pop_callback(NeptuneCallback)
        self.save_model_path = self.out_dir + "/final_model"

        log_param("max learning rate", self.training_args.learning_rate)
        log_param("output_dir", self.out_dir)
        log_param("path_to_training_data", self.path_to_training_data)
        log_param("path_to_eval_data", self.path_to_eval_data)
        log_param("Save model path", self.save_model_path)

    def execute(self) -> None:
        self.trainer.train()
        print(f"Saving model at {self.save_model_path}")
        self.trainer.save_model(self.save_model_path)
