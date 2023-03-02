import math

from jobs.core import Job
from mcts.mcts import expansion_function, score_function
from metric_logging import log_param


class RunMCTSJob(Job):
    def __init__(
        self,
        initial_state,
        time_limit: float = None,
        max_mcts_passes: int = None,
        exploration_constant: float = 1 / math.sqrt(2),
        score=score_function,
        expand=expansion_function,
        out_dir: str = None,
        file_name: str = None,
    ):
        self.initial_state = initial_state
        self.time_limit = time_limit
        self.max_mcts_passes = max_mcts_passes
        self.exploration_constant = exploration_constant
        self.score = score
        self.expand = expand
        self.out_dir = out_dir
        self.file_name = file_name

        log_param("Initial state", str(self.initial_state))
        log_param("Time limit", self.time_limit)
        log_param("Save tree path", f"{self.out_dir}/{self.file_name}")
        log_param("Max number of mcts passes", self.max_mcts_passes)
        log_param("Exploration constant", self.exploration_constant)

    def execute(self):
        import os
        import pickle
        from mcts.mcts import Tree

        tree = Tree(
            initial_state=self.initial_state,
            time_limit=self.time_limit,
            max_mcts_passes=self.max_mcts_passes,
            exploration_constant=self.exploration_constant,
            score=self.score,
            expand=self.expand,
        )
        tree.mcts()

        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir, exist_ok=True)
        with open(f"{self.out_dir}/{self.file_name}", "wb+") as f:
            pickle.dump(tree, f)
