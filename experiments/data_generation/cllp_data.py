from datetime import date
from mrunner.helpers.specification_helper import create_experiments_helper

base_config = {
    "run.job_class": "@jobs.LeelaPrepareAndSaveData",
    "LeelaPrepareAndSaveData.pandas_data_prepare_cls": "@data.CLLPPrepareAndSaveData",


    "CLLPPrepareAndSaveData.data_path": "/leela_generator_data_train",
    "CLLPPrepareAndSaveData.out_path": "/save_data/cllp_data",
    "CLLPPrepareAndSaveData.files_batch_size": 60,
    "CLLPPrepareAndSaveData.p_sample": 0.5,
    "CLLPPrepareAndSaveData.files_limit": 1200,
    "use_neptune": True,
}

params_grid = {
    "idx": [0],
}

experiments_list = create_experiments_helper(
    experiment_name=f"generate_cllp_data",
    project_name="pmtest/subgoal-chess",
    base_config=base_config,
    params_grid=params_grid,
    script="python3 -m runner --mrunner",
    exclude=["data", ".pytest_cache", "out", ".git"],
    python_path="",
    tags=["leela", "generate", "cllp", "data"],
    with_neptune=True,
    env={},
)
