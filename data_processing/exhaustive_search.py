import random
from typing import Union, Tuple, List

import chess
from matplotlib import pyplot as plt

from chess_engines.third_party.stockfish import StockfishEngine
from data_processing.data_utils import immutable_boards_to_img
from data_structures.data_structures import ImmutableBoard


class ExhaustiveSearch:
    def __init__(
        self,
        stockfish_engine: StockfishEngine,
        input_board: ImmutableBoard,
        max_distance: int,
        top_n_moves: Union[int, None],
    ):

        self.stockfish = stockfish_engine

        self.input_board = input_board
        self.depth = max_distance
        self.top_n_moves = top_n_moves

        self.dist_to_states = {i: set() for i in range(max_distance + 1)}
        self.dist_to_states[0].add(input_board)
        self.states_to_dist = {input_board: 0}
        self.boards_to_dist = {input_board.board: 0}

        self.search()

    def moves_to_expand(self, immutable_board: ImmutableBoard) -> Tuple[chess.Move]:
        if self.top_n_moves is None:
            return immutable_board.legal_moves()
        else:
            return self.stockfish.get_top_n_moves(immutable_board, self.top_n_moves)

    def search(self) -> None:
        print(f"Searching for {self.depth} steps with {self.top_n_moves} moves per step")
        for dist in range(1, self.depth + 1):
            for state in self.dist_to_states[dist - 1]:
                for move in self.moves_to_expand(state):
                    new_state = state.act(move)
                    if new_state not in self.states_to_dist:
                        self.dist_to_states[dist].add(new_state)
                        self.states_to_dist[new_state] = dist
                        self.boards_to_dist[new_state.board] = dist

    def get_random_sample(self) -> plt.Figure:
        boards_samples = []
        for i in range(self.depth + 1):
            boards_samples.append(random.sample(list(self.dist_to_states[i]), 1)[0])
        return immutable_boards_to_img(boards_samples, ["dist_" + str(i) for i in range(self.depth + 1)])

    def check_subgoals(self, subgoals: List[ImmutableBoard]) -> dict:
        data = {"accessible": [], "distance": []}
        for subgoal in subgoals:
            if subgoal.board in self.boards_to_dist:
                data["accessible"].append(True)
                data["distance"].append(self.boards_to_dist[subgoal.board])
            else:
                data["accessible"].append(False)
                data["distance"].append(None)
        return data
