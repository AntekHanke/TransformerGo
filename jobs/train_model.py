import os
import pickle
from typing import Type, Optional, List

import numpy as np
import torch
from transformers import (
    Trainer,
    BartForConditionalGeneration,
    BartConfig,
    TrainingArguments,
)
from go_policy.policy_config import AlphaZeroPolicyConfig
from go_policy.policy_model  import AlphaZeroPolicyModel
from transformers.integrations import NeptuneCallback

import metric_logging
from data_processing.pandas_iterable_data_provider import (
    PandasIterableDataProvider,
    PandasIterableSubgoalAllDistancesDataProvider,
)
from data_processing.pandas_static_dataset_provider import (
    PandasStaticDataProvider,
    PandasStaticSubgoalAllDistancesDataProvider,
)
from jobs.core import Job
from metric_logging import (
    log_param,
    source_files_register,
    pytorch_callback_loggers,
    log_object,
    get_experiment_label,
)
from mrunner_utils.mrunner_client import resume_neptune

source_files_register.register(__file__)


class TrainModel(Job):
    def __init__(
        self,
        train_data_provider: Type[PandasIterableDataProvider],
        eval_data_provider: Type[PandasStaticDataProvider],
        path_to_training_data: Optional[str] = None,
        path_to_eval_data: Optional[str] = None,
        files_batch_size: int = None,
        eval_n_batches: int = None,
        range_of_k: Optional[List[int]] = None,
        prob_take_sample: float = 1.0,
        model_config_cls: Optional[Type[BartConfig]] = None,
        training_args_cls: Type[TrainingArguments] = None,
        out_dir: str = None,
    ) -> None:

        assert files_batch_size is not None, "files_batch_size is not set"

        self.path_to_training_data = path_to_training_data
        self.path_to_eval_data = path_to_eval_data
        self.out_dir = out_dir
        self.model_config_cls = model_config_cls
        self.training_args_cls = training_args_cls

        self.training_args = self.get_training_args()

        self.train_data_provider = train_data_provider(
            data_path=self.path_to_training_data,
            files_batch_size=files_batch_size,
            p_sample=prob_take_sample,
            cycle=True,
            name="train",
            range_of_k=range_of_k,
        )
        assert isinstance(
            self.train_data_provider, PandasIterableDataProvider
        ), f"train_data_provider must be PandasIterableDataProvider, got {train_data_provider}"

        if isinstance(self.train_data_provider, PandasIterableSubgoalAllDistancesDataProvider):
            assert (
                self.train_data_provider.range_of_k is not None
            ), "must specify range_of_k for PandasIterableSubgoalAllDistancesDataProvider"
            assert (
                max(self.train_data_provider.range_of_k) <= 9 and min(self.train_data_provider.range_of_k) >= 1
            ), "range_of_k must be in [1, 9]"

        self.eval_data_provider = eval_data_provider(
            data_path=self.path_to_eval_data,
            files_batch_size=files_batch_size,
            p_sample=prob_take_sample,
            eval_num_samples=eval_n_batches * training_args_cls.per_device_eval_batch_size,
            name="eval",
            range_of_k=range_of_k,
        )
        assert isinstance(
            self.eval_data_provider, PandasStaticDataProvider
        ), f"eval_data_provider must be PandasStaticDataProvider, got {eval_data_provider}"

        if isinstance(self.train_data_provider, PandasStaticSubgoalAllDistancesDataProvider):
            assert (
                self.train_data_provider.range_of_k is not None
            ), "must specify range_of_k for PandasIterableSubgoalAllDistancesDataProvider"
            assert (
                max(self.train_data_provider.range_of_k) <= 9 and min(self.train_data_provider.range_of_k) >= 1
            ), "range_of_k must be in [1, 9]"

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

    def get_training_args(self):
        raise NotImplementedError

    def get_model(self):
        raise NotImplementedError

    def train_model(self):
        raise NotImplementedError

    def execute(self) -> None:
        self.train_model()
        log_param("Save model path", self.save_model_path)
        self.trainer.save_model(self.save_model_path)


class TrainModelFromScratch(TrainModel):
    def get_training_args(self):
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir, exist_ok=True)
        self.training_args = self.training_args_cls(output_dir=self.out_dir, ignore_data_skip=True)
        with open(self.training_args.output_dir + "/training_args.pkl", "wb") as f:
            pickle.dump(self.training_args, f)

        neptune_experiment_label = get_experiment_label()
        if neptune_experiment_label is not None:
            with open(self.training_args.output_dir + "/neptune_experiment_label.txt", "w") as f:
                f.write(neptune_experiment_label)
        return self.training_args

    def get_model(self):
        return BartForConditionalGeneration(self.model_config_cls())

    def train_model(self):
        self.trainer.train()

class TrainConvolutionFromScratch(TrainModelFromScratch):
    
    @staticmethod
    def compute_metrics(eval_preds):
        predictions, labels = eval_preds
        predictions, _ = predictions
        return{
            "accuracy": (predictions == np.prod(labels, axis = -1)).astype(np.float32).mean().item(),
        }
    @staticmethod
    def preprocess_logits_for_metrics(logits, labels):
        pred_ids = torch.argmax(logits, dim = -1)
        labels = torch.prod(labels, dim = -1)
        return pred_ids, labels
    
    def get_model(self):
        return AlphaZeroPolicyModel(self.model_config_cls())
    

class ResumeTraining(TrainModel):
    def __init__(self, checkpoint_path, checkpoint_num, **kwargs):
        self.checkpoint_path = checkpoint_path
        self.checkpoint_num = checkpoint_num
        with open(self.checkpoint_path + "/neptune_experiment_label.txt", "r") as f:
            experiment_label = f.read().replace("\n", "")

        resumed_neptune_experiment = resume_neptune(experiment_label)
        metric_logging.register_logger(resumed_neptune_experiment)
        metric_logging.register_pytorch_callback_logger(resumed_neptune_experiment)

        super().__init__(**kwargs)

    def get_training_args(self):
        with open(self.checkpoint_path + "/training_args.pkl", "rb") as f:
            return pickle.load(f)

    def get_model(self):
        return BartForConditionalGeneration.from_pretrained(self.checkpoint_path + f"/checkpoint-{self.checkpoint_num}")

    def train_model(self):
        self.trainer.train(resume_from_checkpoint=self.checkpoint_path + f"/checkpoint-{self.checkpoint_num}")
