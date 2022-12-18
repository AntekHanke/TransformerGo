from mrunner.helpers.specification_helper import create_experiments_helper

base_config = {
    "run.job_class": "@jobs.EvaluateCLLP",
    "EvaluateCLLP.k": [5],
    "EvaluateCLLP.cllp_model": "one_move",
    "EvaluateCLLP.n_subgoals": 3,
    "EvaluateCLLP.cllp_checkpoint": "/leela_models/cllp_all_moves/final_model",
    "EvaluateCLLP.trees_file_path": "/trees",
    "EvaluateCLLP.n_eval_datapoints": 200,
}

params_grid = {
    "idx": [0],
}

experiments_list = create_experiments_helper(
    experiment_name="Quality of CLLP",
    project_name="pmtest/subgoal-chess",
    base_config=base_config,
    params_grid=params_grid,
    script="python3 -m runner --mrunner",
    exclude=["data", ".pytest_cache", "alpacka.egg-info", "out", ".git"],
    python_path="",
    tags=["quality-data"],
    with_neptune=True,
    env={},
)
