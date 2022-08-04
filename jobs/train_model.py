from jobs.core import Job
from transformers import TrainingArguments, Trainer

class TrainModel(Job):
    def __init__(self, model_config, training_config, data_generation_class):
        self.model_config = model_config
        self.training_config = training_config
        self.data_generation_class = data_generation_class

        self.model =

    def run_training(self):
        pass

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)