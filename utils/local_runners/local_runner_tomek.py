from local_runner import local_run

LOCAL_PATH_BINDING = {
    "/ultra_small_data": "/home/tomasz/Research/subgoal_search_chess/assets/small_datasets_for_local_use/lichess_dataset",
    "/out": "/home/tomasz/Research/subgoal_chess_data/local_leela_models",
}

EXPERIMENT_TRAIN_GENERATOR = (
    "/home/tomasz/Research/subgoal_search_chess/experiments/train/generator/ultra_small_model_old.py"
)
EXPERIMENT_CLLP_DATA_MAKE = "/home/tomasz/Research/subgoal_search_chess/experiments/data_generation/cllp_from_leela.py"
EXPERIMENT_TRAIN_CLLP = "/home/tomasz/Research/subgoal_search_chess/experiments/train/cllp/ultra_small_model_old.py"
EXPERIMENT_TRAIN_POLICY = "/home/tomasz/Research/subgoal_search_chess/experiments/train/pgn_policy/ultra_small_model_old.py"
EXPERIMENT_EVAL_CLLP_ALL_MOVES = (
    "/home/tomasz/Research/subgoal_search_chess/experiments/evaluation/cllp/evaluate_cllp.py"
)
EXPERIMENT_EVAL_CLLP_OME_MOVE = (
    "/home/tomasz/Research/subgoal_search_chess/experiments/evaluation/cllp/evaluate_cllp_one_move.py"
)

ULTRA_SMALL_BERT = "/home/tomasz/Research/subgoal_search_chess/experiments/train/bert_for_sequence/ultra_small_model_old.py"
TRAIN_POLICY = "/home/tomasz/Research/subgoal_search_chess/experiments/train/policy/ultra_small_model.py"
TRAIN_GENERATOR = "/home/tomasz/Research/subgoal_search_chess/experiments/train/generator/ultra_small_model.py"

GEN_POLICY_DATA = "/home/tomasz/Research/subgoal_search_chess/experiments/data_generation/policy_data.py"
GEN_CLLP_DATA = "/home/tomasz/Research/subgoal_search_chess/experiments/data_generation/cllp_data.py"

TEST_LC_ZERO = "/home/tomasz/Research/subgoal_search_chess/experiments/debug_jobs/general_debug_job.py"
TEST_DATA_LICHESS = "/home/tomasz/Research/subgoal_search_chess/experiments/temp/lichess_data_gen_local.py"

ULTRA_SMALL_GENERATOR = "/home/tomasz/Research/subgoal_search_chess/experiments/train/generator/ultra_small_model.py"


local_run(TRAIN_POLICY, True, LOCAL_PATH_BINDING)
