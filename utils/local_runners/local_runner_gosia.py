from local_runner import local_run

LOCAL_PATH_BINDING = {
    "/ultra_small_data": "/home/gosia/PycharmProjects/subgoal_search_chess/assets/small_datasets_for_local_use/lichess_dataset",
    "/out": "/home/gosia/dane/subgoal_chess_data/local_leela_models",
}

TRAIN_POLICY = "/home/gosia/PycharmProjects/subgoal_search_chess/experiments/train/policy/ultra_small_model.py"

local_run(TRAIN_POLICY, True, LOCAL_PATH_BINDING)
