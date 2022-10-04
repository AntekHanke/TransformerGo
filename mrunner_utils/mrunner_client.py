"""Parse and return mrunner gin-configures and set-up Neptune.

This is copied from alpacka (with removed ray setup).
"""
import atexit
import datetime
import os
import pickle

import cloudpickle
import neptune.new as neptune


from transformers import TrainerCallback

from configures.global_config import NEPTUNE_PROJECT, NEPTUNE_API_TOKEN
from metric_logging import source_files_register


def get_configuration(spec_path):
    """Get mrunner experiment specification and gin-configures overrides."""
    try:
        with open(spec_path, "rb") as f:
            specification = cloudpickle.load(f)
    except pickle.UnpicklingError:
        with open(spec_path) as f:
            vars_ = {"script": os.path.basename(spec_path)}
            exec(f.read(), vars_)  # pylint: disable=exec-used
            specification = vars_["experiments_list"][0].to_dict()
            print("NOTE: Only the first experiment from the list will be run!")

    parameters = specification["parameters"]
    gin_bindings = []
    for key, value in parameters.items():
        if isinstance(value, str) and not (value[0] == "@" or value[0] == "%"):
            binding = f'{key} = "{value}"'
        else:
            binding = f"{key} = {value}"
        gin_bindings.append(binding)

    return specification, gin_bindings


class NeptunePytorchCallback(TrainerCallback):
    def __init__(self, run):
        self.run = run

    def on_log(self, args, state, control, logs=None, **kwargs):
        _ = logs.pop("total_flos", None)
        if state.is_local_process_zero:
            step = logs["epoch"]
            for metric_name, value in logs.items():
                if metric_name != "epoch":
                    try:
                        self.run[metric_name].log(value=float(value), step=step)
                    except Exception as e:
                        print(f"Exception while logging metric {metric_name} value {value} step {step}")
                        print(f"Exception: {e}")


class NeptuneLogger:
    """Logs to Neptune."""

    def __init__(self, name, tags, **kwargs):
        """Initialize NeptuneLogger with the Neptune experiment."""

        self.run = neptune.init(
            name=name,
            tags=tags,
            project=NEPTUNE_PROJECT,
            api_token=NEPTUNE_API_TOKEN,
            source_files=source_files_register.get(),
        )

        atexit.register(neptune.stop)

    def log_value(self, name, step, value):
        """Logs a scalar to Neptune."""
        self.run[name].log(value=value, step=step)

    def log_value_without_step(self, name, value):
        self.run[name].log(value=value)

    def log_object(self, name, object):
        """Logs an image to Neptune."""
        self.run[name].log(object)

    def log_param(self, name, value):
        self.run[name] = value

    def get_pytorch_callback(self):
        return NeptunePytorchCallback(self.run)


class NeptuneAPITokenException(Exception):
    def __init__(self):
        super().__init__("NEPTUNE_API_TOKEN environment variable is not set!")


def configure_neptune(specification):
    """Configures the Neptune experiment, then returns the Neptune logger."""
    if "NEPTUNE_API_TOKEN" not in os.environ:
        raise NeptuneAPITokenException()

    git_info = specification.get("git_info", None)
    if git_info:
        git_info.commit_date = datetime.datetime.now()
    # Set pwd property with path to experiment.
    # properties = {"pwd": os.environ.get("NEPTUNEPWD", os.getcwd())}

    return NeptuneLogger(
        name=specification["name"],
        tags=specification["tags"],
        git_info=git_info,
    )
