import math
import os
import sys

experiment_dir_path = os.path.dirname(os.path.abspath("__file__"))
sys.path.append(experiment_dir_path)

from mrunner.helpers.specification_helper import create_experiments_helper

STOCKFISH_PATH = "/data_mg/stockfish/stockfish_15_linux_x64/stockfish_15_x64"
EVAL_DATA_DIR = "/data_mg/data/only_immutable_board_eval_data/eval_immutable_boards_lichess.pkl"
OUT_DIR = "/out_models/mcts_vs_stockfish_statistics"
FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

experiment_config = {
    "run.job_class": "@jobs.CompareMCTSWithStockfish",

    "CompareMCTSWithStockfish.time_limit": 90,
    "CompareMCTSWithStockfish.max_mcts_passes": 200,
    "CompareMCTSWithStockfish.exploration_constant": 1 / math.sqrt(2),
    "CompareMCTSWithStockfish.score_function": "@score_functions.score_function",
    "CompareMCTSWithStockfish.expand_function": "@expand_functions.expand_function",
    "CompareMCTSWithStockfish.stockfish_path": STOCKFISH_PATH,
    "CompareMCTSWithStockfish.stockfish_parameters": {},
    "CompareMCTSWithStockfish.eval_data_dir": EVAL_DATA_DIR,
    "CompareMCTSWithStockfish.out_dir": OUT_DIR,
    "CompareMCTSWithStockfish.sample_seed": 0,
    "CompareMCTSWithStockfish.num_boards_to_compare": 1,

    "expand_function.chess_state_expander_class": "@chess_state_expander.ChessStateExpander",
    "expand_function.cllp_num_beams": 32,
    "expand_function.cllp_num_return_sequences": 8,
    "expand_function.generator_num_beams": 16,
    "expand_function.generator_num_subgoals": 8,
    "expand_function.sort_subgoals_by": "highest_total_probability",
    "expand_function.num_top_subgoals": 4,

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
    experiment_name=f"compare_mcts_with_stockfish",
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
