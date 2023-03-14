from local_runner import local_run

LOCAL_PATH_BINDING = {
    "/ultra_small_data": "/home/tomasz/Research/subgoal_search_chess/assets/small_datasets_for_local_use/lichess_dataset",
    "/out_models": "/home/tomasz/Research/subgoal_chess_data/local_leela_models",
    "/subgoal_chess_data_local": "/home/tomasz/Research/subgoal_chess_data",
}


TRAIN_POLICY = "/home/tomasz/Research/subgoal_search_chess/experiments/train/policy/ultra_small_policy.py"
TRAIN_GENERATOR = (
    "/home/tomasz/Research/subgoal_search_chess/experiments/train/generator/ultra_small_generator_from_scratch.py"
)
RESUME_GENERATOR = (
    "/home/tomasz/Research/subgoal_search_chess/experiments/train/generator/ultra_small_generator_resume.py"
)
GENERATE_TREE = "/home/tomasz/Research/subgoal_search_chess/experiments/generate_tree_local.py"


local_run(GENERATE_TREE, True, LOCAL_PATH_BINDING)
