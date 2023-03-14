from typing import Callable, List

import chess
import pandas as pd

from data_structures.data_structures import ImmutableBoard
from jobs.core import Job
from mcts.mcts import TreeNode, Tree
from stockfish import Stockfish


class CompareMCTSWithStockfish(Job):
    def __init__(
        self,
        time_limit: int,
        max_mcts_passes: int,
        exploration_constant: float,
        score_function: Callable[[TreeNode, chess.Color, float], float],
        expand_function: Callable[..., None],
        stockfish_path: str,
        stockfish_parameters: dict,
        eval_data_dir: str,
        out_dir: str,
        sample_seed: int,
        num_boards_to_compare: int,
    ):
        self.time_limit = time_limit
        self.max_mcts_passes = max_mcts_passes
        self.exploration_constant = exploration_constant
        self.score_function = score_function
        self.expand_function = expand_function
        self.stockfish_path = stockfish_path
        self.stockfish_parameters = stockfish_parameters
        self.eval_data_dir = eval_data_dir
        self.out_dir = out_dir
        self.sample_seed = sample_seed
        self.num_boards_to_compare = num_boards_to_compare

    def execute(self):
        eval_data_df = pd.read_pickle(self.eval_data_dir)
        eval_data_df["input_ids"].sample(n=self.num_boards_to_compare, random_state=self.sample_seed)
        list_of_boards: List[ImmutableBoard] = []
        root_stats_list: List[dict] = []
        stockfish = Stockfish(path=self.stockfish_path, parameters=self.stockfish_parameters)

        for board in list_of_boards:
            stockfish.set_fen_position(board.fen())
            tree = Tree(
                initial_state=board,
                time_limit=self.time_limit,
                max_mcts_passes=self.max_mcts_passes,
                exploration_constant=self.exploration_constant,
                score_function=self.score_function,
                expand_function=self.expand_function,
                output_root_values_list=True,
            )
            mcts_output_dict = tree.mcts()
            root_stats_list.append(
                {
                    "Board": board,
                    "Stockfish value": stockfish.get_evaluation(),
                    "MCTS values": mcts_output_dict["root_values_list"],
                }
            )

        root_stats_df = pd.DataFrame.from_records(root_stats_list)
        root_stats_df.to_pickle(self.out_dir)
