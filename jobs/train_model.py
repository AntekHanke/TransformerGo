from data_processing.chess_data_generator import ChessMovesDataGenerator
from jobs.core import Job
from transformers import TrainingArguments, Trainer, BartForConditionalGeneration, BartConfig

import neptune.new as neptune

# run = neptune.init(
#     project="pmtest/subgoal-chess",  # "<WORKSPACE/PROJECT>"
#     api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIwZDRhNWIxNS1hN2RkLTQ1ZjMtOGRmZi02MWI4NGRkZjA5MGMifQ==",  # enables logging without registration
# )
from mrunner_utils.pytorch_neptune_logger import PytorchNeptuneLogger

# neptune_logger = NeptuneLogger(
#     api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIwZDRhNWIxNS1hN2RkLTQ1ZjMtOGRmZi02MWI4NGRkZjA5MGMifQ==",  # replace with your own
#     project="pmtest/subgoal-chess",  # "<WORKSPACE/PROJECT>"
#     tags=["training", "hello"],  # optional
# )

from transformers import TrainerCallback


class PrinterCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        _ = logs.pop("total_flos", None)
        if state.is_local_process_zero:
            print(f"my logs = {logs}")

class TrainModel(Job):
    def __init__(self, model_config=None, training_args=None, data_generation_class=None):
        self.model_config = model_config
        self.training_args = training_args
        self.data_generator = data_generation_class

        self.model = BartForConditionalGeneration(self.model_config)

        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=ChessMovesDataGenerator("/home/tomek/Research/subgoal_chess_data/database.pgn"),
            eval_dataset=ChessMovesDataGenerator("/home/tomek/Research/subgoal_chess_data/database.pgn"),
)
        self.trainer.add_callback(PrinterCallback)

    def execute(self):
        self.trainer.train()


small_config = BartConfig(vocab_size=128, max_position_embeddings=128, encoder_ffn_dim=512, decoder_ffn_dim=512)
training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # total number of training epochs
    per_device_train_batch_size=2,  # batch size per device during training
    per_device_eval_batch_size=2,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='out',            # directory for storing logs
    logging_steps=10,
)

x = TrainModel(small_config, training_args, None)
x.execute()