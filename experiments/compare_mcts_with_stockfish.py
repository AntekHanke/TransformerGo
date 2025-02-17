import math
import os
import random
import sys
from datetime import date

experiment_dir_path = os.path.dirname(os.path.abspath("__file__"))
sys.path.append(experiment_dir_path)

from mrunner.helpers.specification_helper import create_experiments_helper
from configures.global_config import STOCKFISH_PATH

EVAL_DATA_FILE = "/data_mg/data/only_immutable_board_eval_data/eval_immutable_boards_lichess.pkl"
OUT_DIR = f"/out_models/mcts_vs_stockfish_statistics/{date.today()}_{random.randint(0, 100000)}_lichess"

experiment_config = {
    "run.job_class": "@jobs.CompareMCTSWithStockfish",

    "CompareMCTSWithStockfish.time_limit": None,
    "CompareMCTSWithStockfish.max_mcts_passes": 50,
    "CompareMCTSWithStockfish.exploration_constant": 1 / math.sqrt(2),
    "CompareMCTSWithStockfish.score_function": "@score_functions.score_function",
    "CompareMCTSWithStockfish.expand_function_class": "@expand_functions.StandardExpandFunction",
    "CompareMCTSWithStockfish.stockfish_path": STOCKFISH_PATH,
    "CompareMCTSWithStockfish.stockfish_parameters": {},
    "CompareMCTSWithStockfish.eval_data_file": EVAL_DATA_FILE,
    "CompareMCTSWithStockfish.out_dir": OUT_DIR,
    "CompareMCTSWithStockfish.sample_seed": 0,
    "CompareMCTSWithStockfish.num_boards_to_compare": 1000,

    "StandardExpandFunction.chess_state_expander_or_class": "@chess_state_expander.ChessStateExpander",
    "StandardExpandFunction.cllp_num_beams": 2,
    "StandardExpandFunction.cllp_num_return_sequences": 1,
    "StandardExpandFunction.generator_num_beams": 8,
    "StandardExpandFunction.generator_num_subgoals": 4,
    "StandardExpandFunction.sort_subgoals_by": "highest_min_probability",
    "StandardExpandFunction.num_top_subgoals": 4,

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
    experiment_name=f"compare_mcts_with_stockfish_lichess_subgoal",
    project_name="pmtest/subgoal-chess",
    base_config=experiment_config,
    params_grid=params_grid,
    script="python3 -m runner --mrunner",
    exclude=["data", ".pytest_cache", "out", ".git"],
    python_path="",
    tags=["compare", "mcts", "stockfish"],
    with_neptune=True,
    env={},
)
