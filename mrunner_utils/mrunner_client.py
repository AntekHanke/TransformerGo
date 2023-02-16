"""Parse and return mrunner gin-config and set-up Neptune."""

import datetime
import os
import pickle

import cloudpickle
import neptune.new as neptune

from transformers import TrainerCallback

if "NEPTUNE_API_TOKEN" not in os.environ:
    from configures.global_config import NEPTUNE_API_TOKEN
else:
    NEPTUNE_API_TOKEN = os.environ["NEPTUNE_API_TOKEN"]


def get_configuration(spec_path):
    """Get mrunner experiment specification and gin-config overrides."""
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
        if key == "imports":
            for module_str in value:
                binding = f"import {module_str}"
                gin_bindings.append(binding)
            continue

        if isinstance(value, str) and not value[0] in ("@", "%", "{", "(", "["):
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
            step = state.global_step
            for metric_name, value in logs.items():
                if metric_name != "epoch":
                    try:
                        self.run[metric_name].log(value=float(value), step=step)
                    except Exception as e:
                        print(f"Exception while logging metric {metric_name} value {value} step {step}")
                        print(f"Exception: {e}")


class NeptuneLogger:
    """Logs to Neptune."""

    def __init__(self, experiment):
        """Initialize NeptuneLogger with the Neptune experiment."""
        super().__init__()
        self._experiment = experiment

    def log_value(self, name: str, step: int, value: float) -> None:
        """Logs a scalar with steps to Neptune."""
        self._experiment[name].log(value=value, step=step)

    def log_value_without_step(self, name: str, value: float) -> None:
        """Logs a scalar to Neptune."""
        self._experiment[name].log(value)

    def log_object(self, name: str, object) -> None:
        """Logs an objects (for exaple: images) to Neptune."""
        self._experiment[name].log(object)

    def log_param(self, name: str, value) -> None:
        """Logs a param (for example: string) to Neptune."""
        self._experiment[name].log(value)

    def get_experiment_label(self) -> str:
        """Returns the Neptune experiment label."""
        return self._experiment._label

    def get_pytorch_callback(self) -> NeptunePytorchCallback:
        return NeptunePytorchCallback(self._experiment)


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
    properties = {"pwd": os.environ.get("NEPTUNEPWD", os.getcwd())}
    run = neptune.init_run(
        api_token=NEPTUNE_API_TOKEN,
        project=specification["project"],
        name=specification["name"],
        tags=specification["tags"],
    )
    run["job_params"] = specification["parameters"]
    run["path_to_experimant/properties"] = properties
    run["git_info/git_info"] = git_info

    return NeptuneLogger(run)

def resume_neptune(experiment_label):
    """Resumes the Neptune experiment, then returns the Neptune logger."""
    run = neptune.init_run(
        with_id=experiment_label,
        project = "pmtest/subgoal-chess", #TODO: load this from mrunner, not hardcode
    )
    return NeptuneLogger(run)