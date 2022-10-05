import random
from typing import Dict

from data_processing.chess_data_generator import ChessGamesDataGenerator, ChessDataset
from data_processing.chess_tokenizer import ChessTokenizer
from leela.leela_graph_data_loader import LeelaGMLTree


class NodesSelectionRule:
    pass

class SubgoalMCGamesDataGenerator:
    def __init__(self, k, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.k = k

        self.train_data = {}
        self.eval_data = {}

        self.data_constructed = False

    def get_train_set_generator(self) -> ChessDataset:
        assert self.data_constructed, "Data not constructed, call .create_data() first"
        return ChessDataset(self.train_data)

    def get_eval_set_generator(self) -> ChessDataset:
        assert self.data_constructed, "Data not constructed, call .create_data() first"
        return ChessDataset(self.eval_data)


    # def tree_to_datapoints(self, leela_tree: LeelaGMLTree) -> None:
    #