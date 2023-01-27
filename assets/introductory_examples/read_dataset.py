import matplotlib.pyplot as plt
import pandas as pd

from data_processing.chess_tokenizer import ChessTokenizer
from utils.data_utils import immutable_boards_to_img

# Here is how to read a dataset from a file:
train_df = pd.read_pickle("../small_datasets_for_local_use/lichess_dataset/train_small.pkl")
eval_df = pd.read_pickle("../small_datasets_for_local_use/lichess_dataset/eval_small.pkl")

# Show dataset columns:
print(f"Dataset columsns: {[x for x in train_df.columns]}")

"""The dataset contains the data for the tasks: 
-> subgoal generator training
-> conditional low-level policy training
-> policy training
-> value training
"""

# Let's see how the dataset looks like:

# Traing sample for subgoal generator:
idx = 10
input_board_tokenized = train_df.iloc[idx]["input_ids"]
input_board_detokenized = ChessTokenizer.decode_board(input_board_tokenized)
target_board_tokenized = train_df.iloc[idx]["labels"]
target_board_detokenized = ChessTokenizer.decode_board(target_board_tokenized)

fig = immutable_boards_to_img([input_board_detokenized, target_board_detokenized], ["Input board", "Target board"])
plt.show()

#If we want to train policy we need input board and a move. We have a list of tokenized moves between
# input and target board.

move_for_policy = ChessTokenizer.decode_move([train_df.iloc[idx]["moves_between_input_and_target"][0]])
print(f"Move for policy: {move_for_policy}")