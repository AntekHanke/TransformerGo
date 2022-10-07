import random
from typing import Dict, List, Set

import numpy as np
from tqdm import tqdm

from data_processing.chess_data_generator import ChessDataset

# from data_processing.chess_tokenizer import ChessTokenizer
from data_processing.data_utils import immutable_boards_to_img
from data_structures.data_structures import LeelaSubgoal, ImmutableBoard

# from leela.leela_graph_data_loader import LeelaGMLTree
from leela.leela_graph_data_loader import LeelaGMLTree
from metric_logging import log_value


class LeelaTreeInputSelector:
    def select_inputs(self, graph: LeelaGMLTree, p: float) -> Set[int]:
        pass


class RandomLeelaTreeInputSelector(LeelaTreeInputSelector):
    def select_inputs(self, graph: LeelaGMLTree, fraction_of_graph_to_use: float) -> Set[int]:
        N_count = graph.N_count()
        n_samples = int(fraction_of_graph_to_use * len(N_count))
        probs = np.array([x for x in N_count])
        probs = probs / np.sum(probs)
        nodes_list = list(range(len(N_count)))
        selected_nodes = set(np.random.choice(nodes_list, size=n_samples, replace=False, p=probs))
        return selected_nodes


class LeelaTreeTargetSelector:
    def select_subgoals(self, input_node: int, graph: LeelaGMLTree, k: int, n_subgoals: int) -> List[LeelaSubgoal]:
        raise NotImplementedError


class HighestNSelector(LeelaTreeTargetSelector):
    def select_subgoals(self, input_node: int, graph: LeelaGMLTree, k: int, n_subgoals: int) -> List[LeelaSubgoal]:
        k_successors = graph.k_successors(input_node, k)
        return list(sorted(k_successors, key=lambda x: (x.dist_from_input, x.N), reverse=True))[:n_subgoals]


class BFSNodeSelector:
    def find_subgoals(
        self, root: int, graph: LeelaGMLTree, k: int, n_subgoals: int, max_iters: int = 200
    ) -> List[LeelaSubgoal]:

        selector = HighestNSelector()
        queue = [root]
        visited = set()
        subgoals = []
        iters = 0

        while len(queue) > 0 and iters < max_iters:
            iters += 1
            node = queue.pop(0)
            if node not in visited:
                visited.add(node)
                subgoals.extend(selector.select_subgoals(node, graph, k, n_subgoals))
                for subgoal in subgoals:
                    if subgoal.target_idx not in queue:
                        queue.append(subgoal.target_idx)

        print(f"len(subgoals) = {len(subgoals)}")
        return subgoals


class SubgoalMCGamesDataGenerator:
    def __init__(self, k, n_subgoals, fraction_of_graph_to_use, log_samples_limit=200):
        self.k = k
        self.n_subgoals = n_subgoals
        self.fraction_of_graph_to_use = fraction_of_graph_to_use

        # self.input_selector = RandomLeelaTreeInputSelector()
        # self.target_selector = HighestNSelector()
        self.nodes_selector = BFSNodeSelector()

        self.log_samples_limit = log_samples_limit

        self.data = {}
        self.data_constructed = False
        self.N_list = []
        self.processed_trees = 0

        self.logged_samples = 0

    def get_chess_dataset(self) -> ChessDataset:
        assert self.data_constructed, "Data not constructed, call .create_data() first"
        return ChessDataset(self.data)

    def leela_tree_to_datapoints(self, leela_tree: LeelaGMLTree):
        subgoals = self.nodes_selector.find_subgoals(0, leela_tree, self.k, self.n_subgoals)
        if self.logged_samples < self.log_samples_limit:
            self.log_samples(subgoals)
        self.log_subgoals_stats(subgoals)
        return subgoals

    def log_subgoals_stats(self, subgoals: List[LeelaSubgoal]):
        self.N_list.extend([x.N for x in subgoals])
        log_value("N_mean", self.processed_trees, np.mean(self.N_list))

    def log_samples(self, subgoals: List[LeelaSubgoal]) -> None:
        for subgoal in subgoals:
            if random.random() < 2:
                self.log_leela_subgoal(subgoal)

    def log_leela_subgoal(self, leela_subgoal: LeelaSubgoal):
        self.logged_samples += 1
        img = immutable_boards_to_img(
            [
                ImmutableBoard.from_fen_str(leela_subgoal.input_fen),
                ImmutableBoard.from_fen_str(leela_subgoal.target_fen),
            ],
            [f"Input Moves={leela_subgoal.moves} Lvl={leela_subgoal.input_level}", f"Target. N = {leela_subgoal.N} "],
        )
        # img.save(f"/home/tomek/Research/subgoal_search_chess/tmp_{random.randint(0, 10**4)}.png")
        img.savefig(f"/home/tomek/Research/subgoal_search_chess/tmp/{random.randint(0, 10 ** 7)}.png")
