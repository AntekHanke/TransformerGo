import time

import chess
import matplotlib.pyplot as plt

from data_processing.data_utils import immutable_boards_to_img
from data_structures.data_structures import ImmutableBoard
from subgoal_generator.subgoal_generator import BasicChessSubgoalGenerator

g = BasicChessSubgoalGenerator("/local_leela_models/4gpu_generator/subgoals_k=3")
input_b = ImmutableBoard.from_fen_str("2R5/5kpp/4p3/p4p2/3B4/1K5N/4rNPP/8 b - - 0 29")

for _ in range(2):
    time_s = time.time()
    subgoals = g.generate_subgoals(input_b, 4)
    print(f"Time for generation: {time.time() - time_s}")

fig = immutable_boards_to_img([input_b] + subgoals, ["input"] + [f"subgoal {i}" for i in range(len(subgoals))])
plt.show()
