import math
import os
from collections import namedtuple

from data_structures.data_structures import ImmutableBoard
from jobs.core import Job
from mcts.mcts import expand_function, score_function
from metric_logging import log_param

TreeData = namedtuple("TreeData", "tree_as_list best_tree_state")


class RunMCTSJob(Job):
    def __init__(
        self,
        initial_state_fen: str,
        time_limit: float = None,
        max_mcts_passes: int = None,
        exploration_constant: float = 1 / math.sqrt(2),
        score=score_function,
        expand=expand_function,
        out_dir: str = None,
        file_name: str = None,
    ):
        self.initial_state = ImmutableBoard.from_fen_str(initial_state_fen)
        self.time_limit = time_limit
        self.max_mcts_passes = max_mcts_passes
        self.exploration_constant = exploration_constant
        self.score = score
        self.expand = expand
        self.out_dir = out_dir
        self.file_name = file_name

        log_param("Initial state", str(self.initial_state))
        log_param("Time limit", self.time_limit)
        log_param("Save tree path", os.path.join(self.out_dir, self.file_name))
        log_param("Max number of mcts passes", self.max_mcts_passes)
        log_param("Exploration constant", self.exploration_constant)

    def execute(self):
        import os
        import pickle
        from mcts.mcts import Tree
        from mcts.mcts_tree_network import mcts_tree_network

        tree = Tree(
            initial_state=self.initial_state,
            time_limit=self.time_limit,
            max_mcts_passes=self.max_mcts_passes,
            exploration_constant=self.exploration_constant,
            score_function=self.score,
            expand_function=self.expand,
        )
        mcts_output = tree.mcts()
        mcts_tree_network(tree, os.path.join(self.out_dir, self.file_name + ".html"))
        output = TreeData(tree_as_list=tree.to_list(), best_tree_state=mcts_output["best_child"])

        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir, exist_ok=True)
        with open(os.path.join(self.out_dir, self.file_name + ".pkl"), "wb+") as f:
            pickle.dump(output, f)
