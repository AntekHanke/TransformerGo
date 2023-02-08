from mrunner.helpers.specification_helper import create_experiments_helper

base_config = {
    "run.job_class": "@jobs.RetokenizationJob",
    "RetokenizationJob.source_directory": "/net/pr2/projects/plgrid/plgggmum_crl/tomaszo/subgoal_chess_data/data/k_3_eval/engines/eval/",
    "RetokenizationJob.target_directory": "/net/pr2/projects/plgrid/plgggmum_crl/malgorzatarog/subgoal_chess_data/data/k_3_eval/engines/eval/",
    "use_neptune": True,
}

params_grid = {
    "idx": [0],
}

experiments_list = create_experiments_helper(
    experiment_name=f"retokenize-chess",
    project_name="pmtest/subgoal-chess",
    base_config=base_config,
    params_grid=params_grid,
    script="python3 -m runner --mrunner",
    exclude=["data", ".pytest_cache", "out", ".git"],
    python_path="",
    tags=["retokenize", "data"],
    with_neptune=True,
    env={},
)
