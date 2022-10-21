import os
import random
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from data_processing.chess_data_generator import ChessDataset

# from data_processing.chess_tokenizer import ChessTokenizer
from data_processing.chess_tokenizer import ChessTokenizer
from data_processing.data_utils import immutable_boards_to_img, is_fen_game_over

# from data_structures.data_structures import ImmutableBoard

# from leela.leela_graph_data_loader import LeelaGMLTree
from leela.leela_graph_data_loader import LeelaGMLTree, data_trees_generator, LeelaSubgoal
from metric_logging import log_value, log_object, log_param


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
        selected_candidates = list(sorted(k_successors, key=lambda x: (x.dist_from_input, x.N), reverse=True))[
            :n_subgoals
        ]
        return [candidate.to_full_subgoal() for candidate in selected_candidates]


class BFSNodeSelector:
    def find_subgoals(
        self, root: int, graph: LeelaGMLTree, k: int, n_subgoals: int, max_iters: int = 200
    ) -> Tuple[List[LeelaSubgoal], Dict]:

        selector = HighestNSelector()
        queue = [root]
        visited = set()
        subgoals: List[LeelaSubgoal] = []

        iters = 0
        stats = {"game_over_states": 0}

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
            elif subgoal.target_immutable_board.to_board().is_game_over():
                stats["game_over_states"] += 1
                correct_subgoals.append(subgoal)

        return correct_subgoals, stats


def list_of_subgoals_to_df(subgoals: List[LeelaSubgoal]) -> pd.DataFrame:
    leela_subgoal_fields = LeelaSubgoal._fields
    data_batch = {field: [] for field in leela_subgoal_fields}
    data_batch["input_ids"] = []
    data_batch["labels"] = []
    for subgoal in subgoals:
        for field in leela_subgoal_fields:
            data_batch[field].append(getattr(subgoal, field))
        data_batch["input_ids"].append(
            ChessTokenizer.encode_immutable_board(subgoal.input_immutable_board)
            + [ChessTokenizer.vocab_to_tokens["<SEP>"]]
        )
        data_batch["labels"].append(ChessTokenizer.encode_immutable_board(subgoal.target_immutable_board))
    return pd.DataFrame(data_batch)


class SubgoalMCGamesDataGenerator:
    def __init__(
        self,
        k=None,
        n_subgoals=None,
        total_datapoints=None,
        log_samples_limit=200,
        input_data_dir=None,
        input_file_name="all_trees.txt",
        save_data_path=None,
        save_data_every=3000,
        # columns_to_save: Tuple[str] = ('input_ids', 'labels', 'moves'),
    ):
        self.k = k
        self.n_subgoals = n_subgoals
        self.total_datapoints = total_datapoints

        self.log_samples_limit = log_samples_limit
        self.input_data_dir = input_data_dir
        self.input_file_name = input_file_name
        self.save_data_path = save_data_path + f"_k={self.k}.pkl"
        self.save_data_every = save_data_every
        # self.columns_to_save = columns_to_save

        self.nodes_selector = BFSNodeSelector()
        self.data = pd.DataFrame()
        self.data_constructed = False
        self.N_list = []
        self.dist_list = []
        self.dist_count = {i: 0 for i in range(self.k + 1)}
        self.processed_trees = 0
        self.processed_files = 0

        self.logged_samples = 0
        self.paths_to_trees = []

        self.extra_game_over_states = 0

        log_param("save_data_path", self.save_data_path)

    def get_paths(self):
        total_paths = 0
        for dir_data in os.walk(self.input_data_dir):
            if self.input_file_name in dir_data[-1]:
                self.paths_to_trees.append(os.path.join(dir_data[0], self.input_file_name))
                total_paths += 1
        log_value("total_leela_tree_files", 0, total_paths)
        log_param("total_leela_tree_files", total_paths)
        for path in self.paths_to_trees:
            log_object("leela_tree_file", path)
        # print(f"Found paths: {self.paths_to_trees}")

    def generate_data(self):
        self.get_paths()
        for path in tqdm(self.paths_to_trees):
            log_object("path", path)
            if len(self.data) < self.total_datapoints:
                for tree in data_trees_generator(path):
                    self.leela_tree_to_datapoints(tree)
                    if len(self.data) > self.total_datapoints:
                        break
                self.processed_files += 1
                log_value("processed_files", self.processed_files, self.processed_files)
        self.data.to_pickle(self.save_data_path)

    # def get_chess_dataset(self) -> ChessDataset:
    #     assert self.data_constructed, "Data not constructed, call .create_data() first"
    #     return ChessDataset(self.data)

    def leela_tree_to_datapoints(self, leela_tree: LeelaGMLTree):
        subgoals, stats = self.nodes_selector.find_subgoals(0, leela_tree, self.k, self.n_subgoals, 200)
        self.extra_game_over_states += stats["game_over_states"]
        new_data_all = list_of_subgoals_to_df(subgoals)
        # new_data_selected = new_data_all[list(self.columns_to_save)]
        self.data = pd.concat([self.data, new_data_all], ignore_index=True)
        if self.logged_samples < self.log_samples_limit:
            self.log_samples(subgoals)
        self.log_subgoals_stats(subgoals)
        self.processed_trees += 1
        log_value("processed_trees", self.processed_trees, self.processed_trees)
        if self.save_data_path is not None and self.processed_trees % self.save_data_every == 0:
            self.data.to_pickle(self.save_data_path)
        log_value("extra_game_over_states", self.processed_trees, self.extra_game_over_states)
        return subgoals

    def log_subgoals_stats(self, subgoals: List[LeelaSubgoal]):
        for subgoal in subgoals:
            self.N_list.append(subgoal.N)
            self.dist_list.append(subgoal.dist_from_input)
            self.dist_count[subgoal.dist_from_input] += 1

        log_value("Total subgoals from tree", self.processed_trees, len(subgoals))
        log_value("Total subgoals", self.processed_trees, len(self.data))
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
                leela_subgoal.input_immutable_board,
                leela_subgoal.target_immutable_board,
            ],
            [f"Input Moves={leela_subgoal.moves} Lvl={leela_subgoal.input_level}", f"Target. N = {leela_subgoal.N} "],
        )
        log_object("Data sample", img)
