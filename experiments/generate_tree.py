import os
import random
import sys

experiment_dir_path = os.path.dirname(os.path.abspath("__file__"))
sys.path.append(experiment_dir_path)

from datetime import date
from mrunner.helpers.specification_helper import create_experiments_helper

OUT_DIR = f"/out_models/trees/{date.today()}_{random.randint(0,100000)}"
OUT_FILE_NAME = "tree"
FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

experiment_config = {
    "run.job_class": "@jobs.RunMCTSJob",

    "RunMCTSJob.initial_state_fen": FEN,
    "RunMCTSJob.time_limit": None,
    "RunMCTSJob.max_mcts_passes": 50,
    "RunMCTSJob.score_function": "@score_functions.score_function",
    "RunMCTSJob.expand_function_class": "@expand_functions.StandardExpandFunction",
    "RunMCTSJob.out_dir": OUT_DIR,
    "RunMCTSJob.out_file_name": OUT_FILE_NAME,

    "StandardExpandFunction.chess_state_expander_or_class": "@chess_state_expander.ChessStateExpander",
    "StandardExpandFunction.cllp_num_beams": 2,
    "StandardExpandFunction.cllp_num_return_sequences": 1,
    "StandardExpandFunction.generator_num_beams": 8,
    "StandardExpandFunction.generator_num_subgoals": 4,
    "StandardExpandFunction.generator_num_subgoals_first_layer": 6,
    "StandardExpandFunction.subgoal_distance_k": 3,
    "StandardExpandFunction.sort_subgoals_by": "highest_min_probability",
    "StandardExpandFunction.num_top_subgoals": 4,
    "StandardExpandFunction.num_top_subgoals_first_layer": 6,

    "ChessStateExpander.chess_policy_class": "@neural_networks.LCZeroPolicy",
    "ChessStateExpander.chess_value_class": "@neural_networks.LCZeroValue",
    "ChessStateExpander.subgoal_generator_or_class": "@neural_networks.BasicChessSubgoalGenerator",
    "ChessStateExpander.cllp_or_class": "@neural_networks.CLLP",

    "BasicChessSubgoalGenerator.checkpoint_path_or_model": "/data_gg/out_models/medium_generator_from_scratch_with_all_subgoals/16_03_22/k_3/checkpoint-95000/",
    "CLLP.checkpoint_path_or_model": "/data_gg/out_models/medium_cllp_from_scratch_k_form_1_to_9/14_03_23/checkpoint-89500/",

    "use_neptune": True,
    }

params_grid = {
    "idx": [0],
}

experiments_list = create_experiments_helper(
    experiment_name=f"generate_tree",
    project_name="pmtest/subgoal-chess",
    base_config=experiment_config,
    params_grid=params_grid,
    script="python3 -m runner --mrunner",
    exclude=["data", ".pytest_cache", "out", ".git"],
    python_path="",
    tags=["generate", "tree", "mcts"],
    with_neptune=True,
    env={},
)
