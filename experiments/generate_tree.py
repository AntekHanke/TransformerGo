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
    "RunMCTSJob.max_mcts_passes": 20,
    "RunMCTSJob.score_function": "@score_functions.score_function",
    "RunMCTSJob.expand_function": "@expand_functions.expand_function",
    "RunMCTSJob.out_dir": OUT_DIR,
    "RunMCTSJob.out_file_name": OUT_FILE_NAME,

    "expand_function.chess_state_expander_class": "@chess_state_expander.ChessStateExpander",
    "expand_function.cllp_num_beams": 32,
    "expand_function.cllp_num_return_sequences": 8,
    "expand_function.generator_num_beams": 32,
    "expand_function.generator_num_subgoals": 16,
    "expand_function.sort_subgoals_by": "highest_total_probability",
    "expand_function.num_top_subgoals": 8,

    "ChessStateExpander.chess_policy_class": "@neural_networks.LCZeroPolicy",
    "ChessStateExpander.chess_value_class": "@neural_networks.LCZeroValue",
    "ChessStateExpander.subgoal_generator_class": "@neural_networks.BasicChessSubgoalGenerator",
    "ChessStateExpander.cllp_class": "@neural_networks.CLLP",

    "BasicChessSubgoalGenerator.checkpoint_path_or_model": "/data_mg/generators/generator_k_3/checkpoint-221500",
    "CLLP.checkpoint_path_or_model": "/data_mg/cllp/medium",
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
