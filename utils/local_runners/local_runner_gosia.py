from local_runner import local_run

LOCAL_PATH_BINDING = {
    "/out_models": "/home/gosia/dane/subgoal_chess_data/out_models",
    "/data_mg": "/home/gosia/dane/subgoal_chess_data",
}

TRAIN_POLICY = "/home/gosia/PycharmProjects/subgoal_search_chess/experiments/train/policy/ultra_small_policy.py"
GENERATE_TREE = "/home/gosia/PycharmProjects/subgoal_search_chess/experiments/generate_tree.py"

local_run(GENERATE_TREE, True, LOCAL_PATH_BINDING)
