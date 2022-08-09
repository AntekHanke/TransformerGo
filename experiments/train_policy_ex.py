from mrunner.helpers.specification_helper import create_experiments_helper

base_config = {
}

params_grid = {
    'idx': [0] * 1,
}

experiments_list = create_experiments_helper(
    experiment_name='Train policy',
    project_name = "pmtest/subgoal-chess",
    base_config=base_config,
    params_grid=params_grid,
    script='python3 policy_train.py',
    exclude=['data', '.pytest_cache', 'alpacka.egg-info', 'out', '.git'],
    python_path='',
    tags=['train-policy'],
    with_neptune=True,
    env={},
)
