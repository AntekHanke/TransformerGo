import random
from typing import Dict, List, Set

import numpy as np

from data_processing.chess_data_generator import ChessDataset

# from data_processing.chess_tokenizer import ChessTokenizer
from data_structures.data_structures import LeelaSubgoal

# from leela.leela_graph_data_loader import LeelaGMLTree
from leela.leela_graph_data_loader import LeelaGMLTree


class LeelaTreeInputSelector:
    def select_inputs(self, graph: LeelaGMLTree, p: float) -> Set[int]:
        pass


class RandomLeelaTreeInputSelector(LeelaTreeInputSelector):
    def select_inputs(self, graph: LeelaGMLTree, fraction_of_graph_to_use: float) -> Set[int]:
        N_count = graph.N_count()
        n_samples = int(fraction_of_graph_to_use * len(N_count))
        probs = np.array([np.sqrt(x) for x in N_count])
        probs = probs / np.sum(probs)
        nodes_list = list(range(len(N_count)))
        selected_nodes = set(np.random.choice(nodes_list, size=n_samples, replace=False, p=probs))
        return selected_nodes


class LeelaTreeTargetSelector:
    def select_subgoals(self, input_node: int, graph: LeelaGMLTree, n_subgoals: int) -> List[LeelaSubgoal]:
            raise NotImplementedError


class HighestNSelector(LeelaTreeTargetSelector):
    def select_subgoals(self, input_node: int, graph: LeelaGMLTree, n_subgoals: int) -> List[LeelaSubgoal]:
        k_successors = graph.k_successors(input_node, self.n_subgoals)
        return list(sorted(k_successors, key=lambda x: (x.dist_from_input, x.N), reverse=True))#[: self.n_subgoals]


class SubgoalMCGamesDataGenerator:
    def __init__(self, k, n_subgoals, fraction_of_graph_to_use):
        self.k = k
        self.n_subgoals = n_subgoals
        self.fraction_of_graph_to_use = fraction_of_graph_to_use

        self.data = {}
        self.data_constructed = False

        self.input_selector = RandomLeelaTreeInputSelector()
        self.target_selector = HighestNSelector()

    def get_chess_dataset(self) -> ChessDataset:
        assert self.data_constructed, "Data not constructed, call .create_data() first"
        return ChessDataset(self.data)

    def leela_tree_to_datapoints(self, leela_tree: LeelaGMLTree):
        selected_inputs = self.input_selector.select_inputs(leela_tree, self.fraction_of_graph_to_use)
        subgoals = []
        for input_node in selected_inputs:
            subgoals.append(self.target_selector.select_subgoals(input_node, leela_tree, self.n_subgoals))

    def log_subgoals_stats(self, subgoals: List[LeelaSubgoal]):
        for subgoal in subgoals:
            if subgoal in self.data:
                self.data[subgoal] += 1
            else:
                self.data[subgoal] = 1

