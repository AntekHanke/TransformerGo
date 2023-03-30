import logging
import os
from pathlib import Path
from typing import Callable, List, Type

import chess
import pandas as pd
from stockfish import Stockfish

from data_structures.data_structures import ImmutableBoard
from jobs.core import Job
from mcts.mcts import TreeNode, Tree, ExpandFunction


class CompareMCTSWithStockfish(Job):
    def __init__(
        self,
        time_limit: int,
        max_mcts_passes: int,
        exploration_constant: float,
        score_function: Callable[[TreeNode, chess.Color, float], float],
        expand_function_class: Type[ExpandFunction],
        stockfish_path: str,
        stockfish_parameters: dict,
        eval_data_file: str,
        out_dir: str,
        sample_seed: int,
        num_boards_to_compare: int,
    ):
        self.time_limit = time_limit
        self.max_mcts_passes = max_mcts_passes
        self.exploration_constant = exploration_constant
        self.score_function = score_function
        self.expand_function_class = expand_function_class
        self.stockfish_path = stockfish_path
        self.stockfish_parameters = stockfish_parameters
        self.eval_data_file = eval_data_file
        self.out_dir = out_dir
        self.sample_seed = sample_seed
        self.num_boards_to_compare = num_boards_to_compare

    def execute(self):
        Path(self.out_dir).mkdir(parents=True, exist_ok=True)

        eval_data_df = pd.read_pickle(self.eval_data_file)
        sampled_list_of_boards: List[ImmutableBoard] = (
            eval_data_df["immutable_board"].sample(n=self.num_boards_to_compare, random_state=self.sample_seed).tolist()
        )
        stats_list: List[dict] = []

        stockfish = Stockfish(path=self.stockfish_path, parameters=self.stockfish_parameters, depth=25)

        for i, board in enumerate(sampled_list_of_boards):
            if board.to_board().is_game_over():
                logging.warning(f"Board number {i} is terminal")
                continue
            stockfish.set_fen_position(board.fen())
            tree = Tree(
                initial_state=board,
                time_limit=self.time_limit,
                max_mcts_passes=self.max_mcts_passes,
                exploration_constant=self.exploration_constant,
                score_function=self.score_function,
                expand_function_or_class=self.expand_function_class,
                output_root_values_list=True,
            )
            mcts_output_dict = tree.mcts()
            player_score_factor = 1 if tree.root.get_player() == chess.WHITE else -1
            stats_list.append(
                {
                    "board": board,
                    "stockfish_value": stockfish.get_evaluation()["value"],
                    "MCTS_values": [x * player_score_factor for x in mcts_output_dict["root_values_list"]],
                }
            )

            if i % 10 == 9:
                root_stats_df = pd.DataFrame.from_records(stats_list)
                root_stats_df.to_pickle(os.path.join(self.out_dir, f"Comparison_{(i+1)//10}.pkl"))

        root_stats_df = pd.DataFrame.from_records(stats_list)
        root_stats_df.to_pickle(os.path.join(self.out_dir, "Comparison_final.pkl"))
