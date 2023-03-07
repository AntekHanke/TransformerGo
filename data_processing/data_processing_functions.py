from typing import List, Dict

import pandas as pd

from configures.global_config import MAX_MOVES_FOR_CLLP, N_MOVES_HISTORY_FOR_MODEL_INPUT
from data_processing.chess_tokenizer import ChessTokenizer


def subgoal_process_df(df: pd.DataFrame):
    df = df[["input_ids", "labels"]]
    return df.to_dict(orient="records")


def subgoal_all_k_process_df(df: pd.DataFrame, range_of_k: List[int]) -> List[Dict]:
    df_all_k: pd.DataFrame = pd.DataFrame(columns=["input_ids", "labels"])
    for k in range_of_k:
        if f"input_ids_{k}" in df.columns:
            df[f"input_ids_{k}"]: pd.Series = df[f"input_ids_{k}"].apply(
                lambda x: [k] + x
            )
            df_all_k = pd.concat(
                [
                    df_all_k,
                    df[[f"input_ids_{k}", f"labels_{k}"]].rename(
                        columns={f"input_ids_{k}": "input_ids", f"labels_{k}": "labels"}
                    ),
                ],
                ignore_index=True,
            )
    df_all_k = df_all_k.sample(frac=1).reset_index(drop=True)
    return df_all_k.to_dict(orient="records")


def policy_process_df(df: pd.DataFrame):
    df = df[["input_ids", "moves_between_input_and_target"]]
    df = df[df["moves_between_input_and_target"].apply(len) > 0]
    df["labels"] = df["moves_between_input_and_target"].apply(lambda x: [x[0]])
    df.drop(columns=["moves_between_input_and_target"], inplace=True)
    return df.to_dict(orient="records")


def policy_with_history_process_df(df: pd.DataFrame):
    df = df[
        ["input_ids", "all_moves_from_start", "moves_between_input_and_target"]
    ].copy(deep=True)
    df = df[df["moves_between_input_and_target"].apply(len) > 0].copy(deep=True)
    df.loc[:, "input_ids"] = df["input_ids"] + df["all_moves_from_start"].apply(
        lambda x: x[-N_MOVES_HISTORY_FOR_MODEL_INPUT:]
    )
    df.loc[:, "labels"] = df["moves_between_input_and_target"].apply(lambda x: [x[0]])
    df.loc[:, "input_ids"] = df["input_ids"].apply(
        lambda x: x
        + (80 + N_MOVES_HISTORY_FOR_MODEL_INPUT - len(x))
        * [ChessTokenizer.vocab_to_tokens["<PAD>"]]
    )
    df.drop(
        columns=["all_moves_from_start", "moves_between_input_and_target"], inplace=True
    )
    return df.to_dict(orient="records")


def subgoal_to_policy_process_df(df: pd.DataFrame):
    df = df[["input_ids", "moves"]]
    data_list = df.to_dict(orient="records")

    def process_single_datapoint(datapoint):
        return {
            "input_ids": datapoint["input_ids"]
            + [ChessTokenizer.vocab_to_tokens["<SEP>"]],
            "labels": ChessTokenizer.encode(datapoint["moves"][0]),
        }

    data = [
        process_single_datapoint(datapoint)
        for datapoint in data_list
        if len(datapoint["moves"]) > 0
    ]
    return data


def cllp_process_df(df: pd.DataFrame):
    df = df[["input_ids", "moves"]]
    data_list = df.to_dict(orient="records")

    def process_single_datapoint(datapoint):
        moves_encoded = [ChessTokenizer.encode(move)[0] for move in datapoint["moves"]]
        if len(moves_encoded) < MAX_MOVES_FOR_CLLP:
            moves_encoded += [ChessTokenizer.special_vocab_to_tokens["<PAD>"]] * (
                MAX_MOVES_FOR_CLLP - len(moves_encoded)
            )
        return {
            "input_ids": datapoint["input_ids"]
            + [ChessTokenizer.vocab_to_tokens["<SEP>"]]
            + datapoint["labels"]
            + [ChessTokenizer.vocab_to_tokens["<SEP>"]],
            "labels": moves_encoded,
        }

    data = [
        process_single_datapoint(datapoint)
        for datapoint in data_list
        if len(datapoint["moves"]) > 0
    ]
    return data