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
from metric_logging import log_param, source_files_register, pytorch_callback_loggers, log_object
from utils.global_params_handler import GlobalParamsHandler

source_files_register.register(__file__)

# def prepare_datasets(
#     train_data_provider: Type[PandasIterableDataProvider],
#     eval_data_provider: Type[PandasStaticDataProvider],
#     path_to_training_data: str = None,
#     path_to_eval_data: str = None,
#     prob_take_sample: float = 1.0,
#     files_batch_size: int = None,
#     eval_n_batches: int = None,
#     training_args_cls: Type[TrainingArguments] = TrainingArguments,
# ):
#     train_data_provider = train_data_provider(
#         data_path=path_to_training_data,
#         files_batch_size=files_batch_size,
#         p_sample=prob_take_sample,
#         cycle=True,
#         name="train",
#     )
#     assert isinstance(
#         train_data_provider, PandasIterableDataProvider
#     ), f"train_data_provider must be PandasIterableDataProvider, got {train_data_provider}"
#
#     eval_data_provider = eval_data_provider(
#         data_path=path_to_eval_data,
#         files_batch_size=files_batch_size,
#         p_sample=prob_take_sample,
#         eval_num_samples=eval_n_batches * training_args_cls.per_device_eval_batch_size,
#         name="eval",
#     )
#     assert isinstance(
#         eval_data_provider, PandasStaticDataProvider
#     ), f"eval_data_provider must be PandasStaticDataProvider, got {eval_data_provider}"
#
#     return train_data_provider, eval_data_provider

# def prepare_trainer(modl, ):
#     trainer =  Trainer(
#         model=model,
#         args=self.training_args,
#         train_dataset=self.train_data_provider,
#         eval_dataset=self.eval_data_provider,
#         preprocess_logits_for_metrics=self.preprocess_logits_for_metrics,
#         compute_metrics=self.compute_metrics,
#     )
#
#     for callback_logger in pytorch_callback_loggers:
#         self.trainer.add_callback(callback_logger.get_pytorch_callback())
#     self.trainer.pop_callback(NeptuneCallback)

class TrainModel(Job):
    def __init__(
        self,
        train_data_provider: Type[PandasIterableDataProvider],
        eval_data_provider: Type[PandasStaticDataProvider],
        path_to_training_data: Optional[str] = None,
        path_to_eval_data: Optional[str] = None,
        files_batch_size: int = None,
        eval_n_batches: int = None,
        prob_take_sample: float = 1.0,
        model_config_cls: Optional[Type[BartConfig]] = None,
        training_args_cls: Type[TrainingArguments] = None,
        checkpoint_to_resume: Optional[str] = None,
        out_dir: str = None,
    ) -> None:

        assert files_batch_size is not None, "files_batch_size is not set"

        self.path_to_training_data = path_to_training_data
        self.path_to_eval_data = path_to_eval_data
        self.out_dir = out_dir
        self.model_config_cls = model_config_cls
        self.training_args = training_args_cls(output_dir=self.out_dir + "/out")
        self.checkpoint_to_resume = checkpoint_to_resume

        self.train_data_provider = train_data_provider(
            data_path=self.path_to_training_data,
            files_batch_size=files_batch_size,
            p_sample=prob_take_sample,
            cycle=True,
            name="train",
        )
        assert isinstance(
            self.train_data_provider, PandasIterableDataProvider
        ), f"train_data_provider must be PandasIterableDataProvider, got {train_data_provider}"

        self.eval_data_provider = eval_data_provider(
            data_path=self.path_to_eval_data,
            files_batch_size=files_batch_size,
            p_sample=prob_take_sample,
            eval_num_samples=eval_n_batches * training_args_cls.per_device_eval_batch_size,
            name="eval",
        )
        assert isinstance(
            self.eval_data_provider, PandasStaticDataProvider
        ), f"eval_data_provider must be PandasStaticDataProvider, got {eval_data_provider}"

        self.model = self.get_model()

        for param_name, param_value in self.training_args.to_dict().items():
            log_object(f"Training arguments/{param_name}", str(param_value))

        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_data_provider,
            eval_dataset=self.eval_data_provider,
            preprocess_logits_for_metrics=self.preprocess_logits_for_metrics,
            compute_metrics=self.compute_metrics,
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

    @staticmethod
    def compute_metrics(eval_preds):
        predictions, labels = eval_preds
        predictions, _ = predictions
        return {
            "accuracy": (predictions == labels).astype(np.float32).mean().item(),
            "perfect_sequence": (predictions == labels).all(axis=1).astype(np.float32).mean().item(),
        }

    @staticmethod
    def preprocess_logits_for_metrics(logits, labels):
        pred_ids = torch.argmax(logits[0], dim=-1)
        return pred_ids, labels

    def get_model(self, model_config_cls, checkpoint_to_resume):
        raise NotImplementedError

    def train_model(self):
        raise NotImplementedError

    def execute(self) -> None:
        self.train_model()
        log_param("Save model path", self.save_model_path)
        self.trainer.save_model(self.save_model_path)


class TrainModelFromScratch(TrainModel):
    def get_model(self):
        assert self.checkpoint_to_resume is None, "checkpoint_to_resume must be None"
        return BartForConditionalGeneration(self.model_config_cls())

    def train_model(self):
        self.trainer.train()

# class ResumeTraining(TrainModel):
#     def get_model(self, model_config_cls, checkpoint_to_resume):
#         return BartForConditionalGeneration.from_pretrained(checkpoint_to_resume)
#
#     def train_model(self):
#         self.trainer.train()



