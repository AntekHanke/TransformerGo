from transformers import TrainerCallback
import neptune.new as neptune

from config.global_config import NEPTUNE_PROJECT, NEPTUNE_API_TOKEN, source_files_register

class NeptunePytorchCallback(TrainerCallback):
    def __init__(self, run):
        self.run = run

    def on_log(self, args, state, control, logs=None, **kwargs):
        _ = logs.pop("total_flos", None)
        if state.is_local_process_zero:
            step = logs["epoch"]
            for metric_name, value in logs.items():
                if metric_name != "epoch":
                    self.run[metric_name].log(value=value, step=step)

class NeptuneLogger:
    """Logs to Neptune."""

    def __init__(self, name, tags, **kwargs):
        """Initialize NeptuneLogger with the Neptune experiment."""
        self.run = neptune.init(
            name=name,
            tags=tags,
            project=NEPTUNE_PROJECT,
            api_token=NEPTUNE_API_TOKEN,
            source_files=source_files_register.get()
        )

    def log_value(self, name, step, value):
        """Logs a scalar to Neptune."""
        self.run[name].log(value=value, step=step)

    def log_value_without_step(self, name, value):
        self.run[name].log(value=value)

    def log_object(self, name, object):
        """Logs an image to Neptune."""
        self.run[name].log(object)


    def get_pytorch_callback(self):
        return NeptunePytorchCallback(self.run)