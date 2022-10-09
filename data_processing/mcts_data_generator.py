import random
from typing import Dict, List, Set

import numpy as np
import pandas as pd
from tqdm import tqdm

from data_processing.chess_data_generator import ChessDataset

# from data_processing.chess_tokenizer import ChessTokenizer
from data_processing.data_utils import immutable_boards_to_img, is_fen_game_over
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
        subgoals: List[LeelaSubgoal] = []
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

        correct_subgoals = []
        for subgoal in subgoals:
            if subgoal.dist_from_input == k:
                correct_subgoals.append(subgoal)
            elif is_fen_game_over(subgoal.target_fen):
                correct_subgoals.append(subgoal)

        return correct_subgoals

def list_of_subgoals_to_df(subgoals: List[LeelaSubgoal]) -> pd.DataFrame:
    leela_subgoal_fields = LeelaSubgoal._fields
    data_batch = {field: [] for field in leela_subgoal_fields}
    for subgoal in subgoals:
        for field in leela_subgoal_fields:
            data_batch[field].append(getattr(subgoal, field))
    return pd.DataFrame(data_batch)

class SubgoalMCGamesDataGenerator:
    def __init__(self, k, n_subgoals, log_samples_limit=200):
        self.k = k
        self.n_subgoals = n_subgoals

        self.nodes_selector = BFSNodeSelector()
        self.log_samples_limit = log_samples_limit

        self.data = pd.DataFrame()
        self.data_constructed = False
        self.N_list = []
        self.dist_list = []
        self.dist_count = {i: 0 for i in range(self.k + 1)}
        self.processed_trees = 0

        self.logged_samples = 0

    def get_chess_dataset(self) -> ChessDataset:
        assert self.data_constructed, "Data not constructed, call .create_data() first"
        return ChessDataset(self.data)

    def leela_tree_to_datapoints(self, leela_tree: LeelaGMLTree):
        subgoals = self.nodes_selector.find_subgoals(0, leela_tree, self.k, self.n_subgoals, 200)

        self.data = pd.concat([self.data, list_of_subgoals_to_df(subgoals)], ignore_index=True)
        if self.logged_samples < self.log_samples_limit:
            self.log_samples(subgoals)
        self.log_subgoals_stats(subgoals)
        self.processed_trees += 1
        return subgoals

    def log_subgoals_stats(self, subgoals: List[LeelaSubgoal]):
        for subgoal in subgoals:
            self.N_list.append(subgoal.N)
            self.dist_list.append(subgoal.dist_from_input)
            self.dist_count[subgoal.dist_from_input] += 1

        log_value("Total subgoals from tree", self.processed_trees, len(subgoals))
        log_value("Total subgoals", self.processed_trees, len(self.N_list))
        log_value("N_mean", self.processed_trees, np.mean(self.N_list))
        log_value("dist_mean", self.processed_trees, np.mean(self.dist_list))
        for dist, count in self.dist_count.items():
            # log_value(f"dist_{dist}_count", self.processed_trees, count)
            log_value(f"dist_{dist}_count_fraction", self.processed_trees, count / len(self.N_list))

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
        img.savefig(f"/home/tomek/Research/subgoal_search_chess/tmp/{random.randint(0, 10 ** 7)}.png")
