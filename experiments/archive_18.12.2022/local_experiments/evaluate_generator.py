import chess

from data_processing.data_utils import immutable_boards_to_img
from data_structures.data_structures import ImmutableBoard
from subgoal_generator.subgoal_generator import BasicChessSubgoalGenerator

x = BasicChessSubgoalGenerator("/home/tomek/Research/subgoal_chess_data/generator_k_2/out/checkpoint-2000")
b = chess.Board()
b.push(chess.Move.from_uci("g1h3"))


subgoal = x.generate_subgoals(ImmutableBoard.from_board(b))

import matplotlib.pyplot as plt

fig = immutable_boards_to_img([ImmutableBoard.from_board(b), subgoal], ["input", "target"])

plt.show()
# print(subgoal)
