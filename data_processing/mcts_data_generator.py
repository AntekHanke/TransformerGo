import random
from typing import Dict, List

import numpy as np

from data_processing.chess_data_generator import ChessDataset

# from data_processing.chess_tokenizer import ChessTokenizer
from data_structures.data_structures import LeelaSubgoal

# from leela.leela_graph_data_loader import LeelaGMLTree
from leela.leela_graph_data_loader import LeelaGMLTree


class LeelaTreeInputSelector:
    def select_inputs(self, graph: LeelaGMLTree, p: float) -> List[LeelaSubgoal]:
        pass


class RandomLeelaTreeInputSelector(LeelaTreeInputSelector):
    def select_inputs(self, graph: LeelaGMLTree, p: float) -> List[int]:
        N_count = graph.N_count()
        n_samples = int(p * len(N_count))
        probs = np.array([np.sqrt(x) for x in N_count])
        probs = probs / np.sum(probs)
        nodes_list = list(range(len(N_count)))
        selected_nodes = list(np.random.choice(nodes_list, size=n_samples, replace=False, p=probs))
        return selected_nodes


class LeelaTreeTargetSelector:
    def select_subgoals(self, k_successors: List[LeelaSubgoal]) -> List[LeelaSubgoal]:
        raise NotImplementedError


class HighestNSelector(LeelaTreeTargetSelector):
    def __init__(self, n_subgoals):
        self.n_subgoals = n_subgoals

    def select_subgoals(self, k_successors: List[LeelaSubgoal]) -> List[LeelaSubgoal]:
        return list(sorted(k_successors, key=lambda x: (x.dist_from_input, x.N), reverse=True))#[: self.n_subgoals]


class SubgoalMCGamesDataGenerator:
    def __init__(self, k, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.k = k

        self.data = {}

        self.data_constructed = False

    def get_chess_dataset(self) -> ChessDataset:
        assert self.data_constructed, "Data not constructed, call .create_data() first"
        return ChessDataset(self.data)
