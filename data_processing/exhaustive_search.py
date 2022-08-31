import random

import chess
import chess.engine

from data_processing.data_utils import immutable_boards_to_img
from data_structures.data_structures import ImmutableBoard


class ExhaustiveSearch:
    def __init__(self, input_board: ImmutableBoard, depth: int):
        self.input_board = input_board
        self.depth = depth
        self.dist_to_states = {i: set() for i in range(depth + 1)}
        self.dist_to_states[0].add(input_board)
        self.states_to_dist = {input_board: 0}
        self.search()

    def search(self):
        for dist in range(1, self.depth + 1):
            for state in self.dist_to_states[dist - 1]:
                for move in state.to_board().legal_moves:
                    new_state = state.act(move)
                    if new_state not in self.states_to_dist:
                        self.dist_to_states[dist].add(new_state)
                        self.states_to_dist[new_state] = dist

    def get_random_sample(self):
        boards_samples = []
        for i in range(self.depth+1):
            boards_samples.append(random.sample(self.dist_to_states[i], 1)[0])
        return immutable_boards_to_img(boards_samples, ["dist_" + str(i) for i in range(self.depth+1)])

# # dupa = ExhaustiveSearch(ImmutableBoard.from_board(chess.Board()), 3)
# # samples = dupa.get_random_sample()
# # plt.show()
#
# immutable_board = ImmutableBoard.from_board(chess.Board())
#
# engine = chess.engine.SimpleEngine.popen_uci("stockfish")
#
# moves = immutable_board.legal_moves()
# print(moves)
#
#
# result = engine.analyse(immutable_board.to_board(), chess.engine.Limit(depth=5, ), game=object(), root_moves=moves)
#
# print(result)
#
# engine.quit()