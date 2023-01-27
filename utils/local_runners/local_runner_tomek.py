from local_runner import local_run

LOCAL_PATH_BINDING = {
    "/leela_generator_data_train": "/home/tomasz/Research/subgoal_chess_data/cllp_leela_datasets/prepare",
    "/leela_generator_data_eval": "/home/tomasz/Research/subgoal_chess_data/generator_leela_datasets/train",
    "/leela_cllp_data": "/home/tomasz/Research/subgoal_chess_data/cllp_leela_datasets",
    "/leela_models": "/home/tomasz/Research/subgoal_chess_data/local_leela_models",
    "/pgn": "/home/tomasz/Research/subgoal_chess_data/pgn",
    "/trees": "/home/tomasz/Research/subgoal_chess_data/leela_eval_trees",
    "/leela_bert_data": "/home/tomasz/Research/subgoal_chess_data/generator_leela_datasets",
    "/save_data": "/home/tomasz/Research/subgoal_chess_data/cllp_datasets",
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



local_run(TRAIN_POLICY, False, None)
