from datetime import date
from mrunner.helpers.specification_helper import create_experiments_helper

base_config = {
    "run.job_class": "@jobs.LeelaPrepareAndSaveData",
    "LeelaPrepareAndSaveData.pandas_data_prepare_cls": "@data.PandasPolicyPrepareAndSaveData",


    "PandasPolicyPrepareAndSaveData.data_path": "/leela_generator_data_train/subgoals_k=1",
    "PandasPolicyPrepareAndSaveData.out_path": "/save_data/policy_data/train",
    "PandasPolicyPrepareAndSaveData.files_batch_size": 10,

    "use_neptune": True,
}

params_grid = {
    "idx": [0],
}

experiments_list = create_experiments_helper(
    experiment_name=f"generate_policy_data",
    project_name="pmtest/subgoal-chess",
    base_config=base_config,
    params_grid=params_grid,
    script="python3 -m runner --mrunner",
    exclude=["data", ".pytest_cache", "out", ".git"],
    python_path="",
    tags=["leela", "generate", "policy", "data"],
    with_neptune=True,
    env={},
)
