import pandas as pd


# Here is how to read a dataset from a file:
train_df = pd.read_pickle("../small_datasets_for_local_use/lichess_dataset/train_small.pkl")
eval_df = pd.read_pickle("../small_datasets_for_local_use/lichess_dataset/eval_small.pkl")

# Show dataset columns:
print(train_df.columns)
