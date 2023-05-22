from typing import Optional

import gin

import metric_logging
from mrunner_utils import mrunner_client
from mrunner_utils.mrunner_client import get_configuration
from runner import run

import configures.gin_configurable_classes  # keep this import


def local_run(experiment_path: str, use_neptune: bool, local_path_bindings: Optional[dict] = None):
    """Runs experiment locally, replacing paths in gin_bindings with local paths."""

    specification, gin_bindings = get_configuration(experiment_path)
    corrected_bindings = set()
    for binding in gin_bindings:
        keep_unchanged = True
        if local_path_bindings:
            for general_path, local_path in local_path_bindings.items():
                if general_path in binding:
                    corrected_bindings.add(binding.replace(general_path, local_path))
                    keep_unchanged = False
                    print(f"Corrected path for local execution: {binding} -> {corrected_bindings}")
            if keep_unchanged:
                corrected_bindings.add(binding)
        else:
            corrected_bindings = gin_bindings

    corrected_bindings = list(corrected_bindings)
    print(f"gin_bindings: {corrected_bindings}")

    if use_neptune:
        neptune_logger = mrunner_client.configure_neptune(specification)
        metric_logging.register_logger(neptune_logger)
        metric_logging.register_pytorch_callback_logger(neptune_logger)

    gin.parse_config_files_and_bindings(None, corrected_bindings)
    run()
