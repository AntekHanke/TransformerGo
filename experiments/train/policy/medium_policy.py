import os
import sys

experiment_dir_path = os.path.dirname(os.path.abspath('__file__'))
sys.path.append(experiment_dir_path)

from datetime import date
from mrunner.helpers.specification_helper import create_experiments_helper
from experiments.train.policy.common_training_params import medium_model_config

base_config = {
    "TrainModel.train_data_provider": "@data.PandasIterablePolicyDataProvider",
    "TrainModel.eval_data_provider": "@data.PandasStaticPolicyDataProvider",
    "TrainModel.out_dir": f"/models/policy/medium_model/no_history/{date.today()}",
}

params_grid = {
    "idx": [0],
}

experiments_list = create_experiments_helper(
    experiment_name=f"medium-policy-model",
    project_name="pmtest/subgoal-chess",
    base_config=dict(medium_model_config, **base_config),
    params_grid=params_grid,
    script="python3 -m runner --mrunner",
    exclude=["data", ".pytest_cache", "out", ".git"],
    python_path="",
    tags=["policy", "train", "medium"],
    with_neptune=True,
    env={},
)
