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

def prepare_datasets(
    train_data_provider: Type[PandasIterableDataProvider],
    eval_data_provider: Type[PandasStaticDataProvider],
    path_to_training_data: str = None,
    path_to_eval_data: str = None,
    prob_take_sample: float = 1.0,
    files_batch_size: int = None,
    eval_n_batches: int = None,
    training_args_cls: Type[TrainingArguments] = TrainingArguments,
):
    train_data_provider = train_data_provider(
        data_path=path_to_training_data,
        files_batch_size=files_batch_size,
        p_sample=prob_take_sample,
        cycle=True,
        name="train",
    )
    assert isinstance(
        train_data_provider, PandasIterableDataProvider
    ), f"train_data_provider must be PandasIterableDataProvider, got {train_data_provider}"

    eval_data_provider = eval_data_provider(
        data_path=path_to_eval_data,
        files_batch_size=files_batch_size,
        p_sample=prob_take_sample,
        eval_num_samples=eval_n_batches * training_args_cls.per_device_eval_batch_size,
        name="eval",
    )
    assert isinstance(
        eval_data_provider, PandasStaticDataProvider
    ), f"eval_data_provider must be PandasStaticDataProvider, got {eval_data_provider}"

    return train_data_provider, eval_data_provider

def prepare_trainer()

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
        model_config_cls: Type[BartConfig] = None,
        training_args_cls: Type[TrainingArguments] = None,
        checkpoint_to_resume: Optional[str] = None,
        out_dir: str = None,
    ) -> None:

        assert files_batch_size is not None, "files_batch_size is not set"
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

        self.checkpoint_to_resume = checkpoint_to_resume

        self.training_args = training_args_cls(output_dir=self.out_dir + "/out")
        if global_params_handler.learning_rate is not None:
            self.training_args.learning_rate = global_params_handler.learning_rate

        self.model_config = model_config_cls()
        self.model = BartForConditionalGeneration(self.model_config)

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

    def execute(self) -> None:
        if self.checkpoint_to_resume is not None:
            self.trainer.train(resume_from_checkpoint=self.checkpoint_to_resume)
        else:
            self.trainer.train()
        print(f"Saving model at {self.save_model_path}")
        self.trainer.save_model(self.save_model_path)


class ResumeTraining(Job):
    def __init__(self, checkpoint_to_resume: str, out_dir: str) -> None:
        self.dir_with_checkpoints = None
        self.checkpoint_numer = None
        self.model = BartForConditionalGeneration.from_pretrained(self.checkpoint_to_resume)


