from mrunner.helpers.specification_helper import create_experiments_helper

base_config = {
    "run.job_class": "@jobs.LeelaCCLPDataProcessing",
    "LeelaCCLPDataProcessing.pandas_data_provider_class": "@data.PandasCLLPDataGenerator",
    "PandasCLLPDataGenerator.data_path": "/leela_generator_data",
    "PandasCLLPDataGenerator.save_final_df_path": "/leela_cllp_data/cllp_one_move.pkl",
    "PandasCLLPDataGenerator.use_one_move": True,

    "use_neptune": True,
}

params_grid = {"idx": [0]}

experiments_list = create_experiments_helper(
    experiment_name="cllp form Leela one move",
    project_name="pmtest/subgoal-chess",
    base_config=base_config,
    params_grid=params_grid,
    script="python3 -m runner --mrunner",
    exclude=["data", ".pytest_cache", "alpacka.egg-info", "out", ".git"],
    python_path="",
    tags=["subgoals_from_leela"],
    with_neptune=True,
    env={},
)
