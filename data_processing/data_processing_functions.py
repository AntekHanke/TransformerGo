import pandas as pd

from configures.global_config import MAX_MOVES_FOR_CLLP
from data_processing.chess_tokenizer import ChessTokenizer


def subgoal_process_df(df: pd.DataFrame):
    df = df[["input_ids", "labels"]]
    return df.to_dict(orient="records")


def policy_process_df(df: pd.DataFrame):
    df = df[["input_ids", "moves_between_input_and_target"]]
    df = df[df["moves_between_input_and_target"].apply(len) > 0]
    df["labels"] = df["moves_between_input_and_target"].apply(lambda x: [x[0]])
    df.drop(columns=["moves_between_input_and_target"], inplace=True)
    return df.to_dict(orient="records")


def policy_with_history_process_df(df: pd.DataFrame):
    df = df[["input_ids", "all_moves_from_start", "moves_between_input_and_target"]]
    df["input_ids"] = df["input_ids"] + df["all_moves_from_start"]
    df.drop(columns=["all_moves_from_start"], inplace=True)
    return policy_added_history_common_processing(df)


def policy_only_history_process_df(df: pd.DataFrame):
    df = df[["all_moves_from_start", "moves_between_input_and_target"]]
    df.rename(columns={"all_moves_from_start": "input_ids"}, inplace=True)
    return policy_added_history_common_processing(df)


def policy_added_history_common_processing(df: pd.DataFrame):
    df = df[df["moves_between_input_and_target"].apply(len) > 0]
    df["labels"] = df["moves_between_input_and_target"].apply(lambda x: [x[0]])
    df["input_ids"].apply(lambda x: x.append(ChessTokenizer.vocab_to_tokens["<SEP>"]))
    df.drop(columns=["moves_between_input_and_target"], inplace=True)
    return df.to_dict(orient="records")

def subgoal_to_policy_process_df(df: pd.DataFrame):
    df = df[["input_ids", "moves"]]
    data_list = df.to_dict(orient="records")

    def process_single_datapoint(datapoint):
        return {
            "input_ids": datapoint["input_ids"] + [ChessTokenizer.vocab_to_tokens["<SEP>"]],
            "labels": ChessTokenizer.encode(datapoint["moves"][0]),
        }

    data = [process_single_datapoint(datapoint) for datapoint in data_list if len(datapoint["moves"]) > 0]
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

    data = [process_single_datapoint(datapoint) for datapoint in data_list if len(datapoint["moves"]) > 0]
    return data
