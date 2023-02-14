from mrunner.helpers.specification_helper import create_experiments_helper
from common_training_params import ultra_small_model_config

base_config = {
    "TrainModel.train_data_provider": "@data.PandasIterablePolicyDataProvider",
    "TrainModel.eval_data_provider": "@data.PandasStaticPolicyDataProvider",
    "TrainModel.out_dir": "/out/policy/ultra_small/no_history",
}

params_grid = {
    "idx": [0],
}

experiments_list = create_experiments_helper(
    experiment_name=f"ultra-small-policy-model",
    project_name="pmtest/subgoal-chess",
    base_config=dict(ultra_small_model_config, **base_config),
    params_grid=params_grid,
    script="python3 -m runner --mrunner",
    exclude=["data", ".pytest_cache", "out", ".git"],
    python_path="",
    tags=["policy", "train", "ultra-small"],
    with_neptune=True,
    env={},
)
