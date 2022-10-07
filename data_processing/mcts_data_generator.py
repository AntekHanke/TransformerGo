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
    def __init__(self, root: int, graph: LeelaGMLTree, k: int, n_subgoals: int):
        self.root = root
        self.graph = graph
        self.k = k
        self.n_subgoals = n_subgoals

        self.selector = HighestNSelector()

    def find_subgoals(self) -> List[LeelaSubgoal]:
        self.queue = [self.root]
        self.visited = set()
        subgoals = []

        while len(self.queue) > 0:
            node = self.queue.pop(0)
            self.visited.add(node)
            subgoals = self.selector.select_subgoals(node, self.graph, self.k, self.n_subgoals)
            subgoals.extend(subgoals)
            for subgoal in subgoals:
                self.queue.append(subgoal.target_idx)




class SubgoalMCGamesDataGenerator:
    def __init__(self, k, n_subgoals, fraction_of_graph_to_use, log_samples_limit=50):
        self.k = k
        self.n_subgoals = n_subgoals
        self.fraction_of_graph_to_use = fraction_of_graph_to_use

        self.input_selector = RandomLeelaTreeInputSelector()
        self.target_selector = HighestNSelector()
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
        selected_inputs = self.input_selector.select_inputs(leela_tree, self.fraction_of_graph_to_use)
        subgoals = []
        for input_node in selected_inputs:
            subgoals.extend(self.target_selector.select_subgoals(input_node, leela_tree, self.k, self.n_subgoals))
        self.processed_trees += 1
        if self.logged_samples < self.log_samples_limit:
            self.log_samples(subgoals)
        self.log_subgoals_stats(subgoals)
        return subgoals

    def log_subgoals_stats(self, subgoals: List[LeelaSubgoal]):
        self.N_list.extend([x.N for x in subgoals])
        log_value("N_mean", self.processed_trees, np.mean(self.N_list))

    def log_samples(self, subgoals: List[LeelaSubgoal]) -> None:
        for subgoal in subgoals:
            if random.random() < 0.1:
                self.log_leela_subgoal(subgoal)

    def log_leela_subgoal(self, leela_subgoal: LeelaSubgoal):
        self.logged_samples += 1
        img = immutable_boards_to_img(
            [
                ImmutableBoard.from_fen_str(leela_subgoal.input_board),
                ImmutableBoard.from_fen_str(leela_subgoal.target_board),
            ],
            [f"Input Moves={leela_subgoal.moves} Lvl={leela_subgoal.input_level}", f"Target. N = {leela_subgoal.N} "],
        )
        # img.save(f"/home/tomek/Research/subgoal_search_chess/tmp_{random.randint(0, 10**4)}.png")
        img.savefig(f"/home/tomek/Research/subgoal_search_chess/tmp/{random.randint(0, 10 ** 4)}.png")
