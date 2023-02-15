from local_runner import local_run

LOCAL_PATH_BINDING = {
    "/ultra_small_data": "/home/tomasz/Research/subgoal_search_chess/assets/small_datasets_for_local_use/lichess_dataset",
    "/out_models": "/home/tomasz/Research/subgoal_chess_data/local_leela_models",
}


TRAIN_POLICY = "/home/tomasz/Research/subgoal_search_chess/experiments/train/policy/ultra_small_policy.py"
TRAIN_GENERATOR = "/home/tomasz/Research/subgoal_search_chess/experiments/train/generator/ultra_small_generator_from_scratch.py"
RESUME_GENERATOR = "/home/tomasz/Research/subgoal_search_chess/experiments/train/generator/resume_ultra_small_generator.py"


local_run(TRAIN_GENERATOR, True, LOCAL_PATH_BINDING)
