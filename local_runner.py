import gin

import metric_logging
from mrunner_utils import mrunner_client
from mrunner_utils.mrunner_client import get_configuration
from runner import run

import configures.gin_configurable_classes  # keep this import

EXPERIMENT_PATH = "/home/tomasz/Research/subgoal_search_chess/experiments/train/generator/ultra_small_model.py"
USE_NEPTUNE = True

LOCAL_PATH_BINDING = {
    "/leela_data_processed": "/home/tomasz/Research/subgoal_chess_data/local_leela_datasets",
    "/leela_models": "/home/tomasz/Research/subgoal_chess_data/local_leela_models",
}


specification, gin_bindings = get_configuration(EXPERIMENT_PATH)
corrected_bindings = []
for binding in gin_bindings:
    for general_path, local_path in LOCAL_PATH_BINDING.items():
        binding = binding.replace(general_path, local_path)
        corrected_bindings.append(binding)
print(f"specification: {specification}")
print(f"gin_bindings: {corrected_bindings}")

if USE_NEPTUNE:
    neptune_logger = mrunner_client.configure_neptune(specification)
    metric_logging.register_logger(neptune_logger)
    metric_logging.register_pytorch_callback_logger(neptune_logger)

gin.parse_config_files_and_bindings(None, corrected_bindings)
run()
