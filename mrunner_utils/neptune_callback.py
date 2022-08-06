from transformers import TrainerCallback
import neptune.new as neptune

from config.global_config import NEPTUNE_PROJECT, NEPTUNE_API_TOKEN, source_files_register


class NeptunePytorchCallback(TrainerCallback):
    def __init__(self):
        self.run = neptune.init(
            project=NEPTUNE_PROJECT,
            api_token=NEPTUNE_API_TOKEN,
            source_files=source_files_register.get()
        )

    def on_log(self, args, state, control, logs=None, **kwargs):
        _ = logs.pop("total_flos", None)
        if state.is_local_process_zero:
            step = logs["epoch"]
            for metric_name, value in logs.items():
                if metric_name != "epoch":
                    self.run[metric_name].log(value=value, step=step)
