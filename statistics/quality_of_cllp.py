import os

import chess
import pandas as pd
from tqdm import tqdm

from configures.global_config import MAX_MOVES_FOR_CLLP
from data_structures.data_structures import ImmutableBoard
from policy.cllp import CLLP


def check_cllp_quality(cllp_checkpoint, data_path, examples_per_k=100):
    file_names_queue = {i: [] for i in range(1, MAX_MOVES_FOR_CLLP + 1)}

    for k in range(1, MAX_MOVES_FOR_CLLP + 1):
        for folder_name in tqdm(os.listdir(data_path + f"/subgoal_{k}")):
            path: str = data_path + f"/subgoal_{k}" + "/" + str(folder_name)
            for file_name in os.listdir(path):
                path_to_file: str = os.path.join(path, file_name)
                if os.path.isfile(path_to_file):
                    file_names_queue[k].append(path_to_file)

    examples = {i: [] for i in range(1, MAX_MOVES_FOR_CLLP + 1)}
    for key, file_names in file_names_queue.items():
        for file_name in file_names:
            df = pd.read_pickle(file_name)
            df = df.sample(n=examples_per_k)
            for row in df.itertuples():
                sample = {
                    "input_board": row.input_immutable_board,
                    "target": row.target_immutable_board,
                    "moves": row.moves,
                }
                examples[key].append(sample)
            break

    cllp = CLLP(cllp_checkpoint, num_return_sequences=8)
    for k, samples in examples.items():
        if k != 3:
            continue

        identical = 0
        good = 0
        trials = 0
        for sample in tqdm(samples):
            paths = cllp.get_path(sample["input_board"], sample["target"])

            trials += 1
            if sample["moves"] in paths:
                identical += 1
            else:
                print(f"K: {k}, paths: {paths}, moves: {sample['moves']}")

            for path in paths:
                try:
                    board = sample["input_board"].to_board()
                    for move in path:
                        board.push(chess.Move.from_uci(move))
                    final = ImmutableBoard.from_board(board)
                    if final == sample["target"]:
                        good += 1
                        break
                except:
                    print("not good")

        print(
            f"K: {k}, success: {identical}, trials: {trials}, success rate: {identical / trials} | good: {good}, good rate: {good / trials}"
        )


check_cllp_quality(
    "/home/tomasz/Research/subgoal_chess_data/local_leela_models/cllp/medium",
    "/home/tomasz/Research/subgoal_chess_data/cllp_leela_datasets/prepare",
    100,
)
