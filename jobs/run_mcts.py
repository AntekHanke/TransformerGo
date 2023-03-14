import math
import os
import pickle
from collections import namedtuple
from pathlib import Path
from typing import Callable

import chess

from data_structures.data_structures import ImmutableBoard
from jobs.core import Job
from mcts.mcts import Tree
from mcts.mcts import expand_function, score_function, TreeNode
from mcts.mcts_tree_network import mcts_tree_network
from metric_logging import log_param

TreeData = namedtuple("TreeData", "tree_as_list best_tree_state")


class RunMCTSJob(Job):
    def __init__(
        self,
        initial_state_fen: str,
        time_limit: float = None,
        max_mcts_passes: int = None,
        exploration_constant: float = 1 / math.sqrt(2),
        score_function: Callable[[TreeNode, chess.Color, float], float] = score_function,
        expand_function: Callable[..., None] = expand_function,
        out_dir: str = None,
        out_file_name: str = None,
    ):
        self.initial_state = ImmutableBoard.from_fen_str(initial_state_fen)
        self.time_limit = time_limit
        self.max_mcts_passes = max_mcts_passes
        self.exploration_constant = exploration_constant
        self.score_function = score_function
        self.expand_function = expand_function
        self.out_dir = out_dir
        self.out_file_name = out_file_name

        log_param("Initial state", str(self.initial_state))
        log_param("Time limit", self.time_limit)
        log_param("Save tree path", os.path.join(self.out_dir, self.out_file_name))
        log_param("Max number of mcts passes", self.max_mcts_passes)
        log_param("Exploration constant", self.exploration_constant)

    def execute(self):
        Path(self.out_dir).mkdir(parents=True, exist_ok=True)

        tree = Tree(
            initial_state=self.initial_state,
            time_limit=self.time_limit,
            max_mcts_passes=self.max_mcts_passes,
            exploration_constant=self.exploration_constant,
            score_function=self.score_function,
            expand_function=self.expand_function,
        )
        mcts_output = tree.mcts()
        mcts_tree_network(tree=tree, target_path=self.out_dir, target_name=self.out_file_name, with_images=True)
        output = TreeData(tree_as_list=tree.to_list(), best_tree_state=mcts_output)
        with open(os.path.join(self.out_dir, self.out_file_name + ".pkl"), "wb+") as f:
            pickle.dump(output, f)
