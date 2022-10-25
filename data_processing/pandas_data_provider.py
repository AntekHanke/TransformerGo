import os
from typing import Dict, List

import pandas as pd

from data_processing.chess_data_generator import ChessDataProvider, ChessDataset
from data_processing.chess_tokenizer import ChessTokenizer
from metric_logging import log_param
from utils.global_params_handler import GlobalParamsHandler


class PandasSubgoalDataProvider(ChessDataProvider):
    def __init__(self, data_path=None, eval_datapoints: int = 10000):
        if data_path is None:
            data_path = GlobalParamsHandler().get_data_path()
            print(f"Data path: {data_path}")

        df = pd.read_pickle(data_path)
        processed_df = self.process_df(df)
        self.data_train = self.pandas_to_dict(processed_df.head(-eval_datapoints))
        self.data_eval = self.pandas_to_dict(processed_df.tail(eval_datapoints))

        log_param("Train set size", len(self.data_train))
        log_param("Eval set size", len(self.data_eval))

    def process_df(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[["input_ids", "labels"]]

    def pandas_to_dict(self, df: pd.DataFrame) -> Dict:
        return df.to_dict(orient="records")

    def get_train_set_generator(self) -> ChessDataset:
        return ChessDataset(self.data_train)

    def get_eval_set_generator(self) -> ChessDataset:
        return ChessDataset(self.data_eval)


class PandasCLLPDataProvider(ChessDataProvider):
    def __init__(self, data_path: str, eval_datapoints: int = 10000):
        if data_path is None:
            data_path = GlobalParamsHandler().get_data_path()

        self.paths_to_data = []
        for dir_data in os.walk(data_path):
            print(dir_data)
            for file in dir_data[-1]:
                if ".pkl" in file:
                    self.paths_to_data.append(os.path.join(dir_data[0], file))

        self.data_path = data_path
        print(self.paths_to_data)
        # df = pd.read_pickle(self.paths_to_data[0])
        # processed_df = self.process_df(df)
        f = 3

    def process_df(self, df: pd.DataFrame) -> pd.DataFrame:
        processed_df = df[["input_ids", "labels", "moves"]]
        processed_df.rename(columns={"labels": "subgoal_board", "input_ids": "input_board"}, inplace=True)

        def tokenize_moves(moves: List[str]) -> List[int]:
            tokenized_moves = []
            for move in moves:
                tokenized_moves.extend(
                    ChessTokenizer.encode_leela_move(move) + [ChessTokenizer.special_vocab_to_tokens["<SEP>"]]
                )
            return tokenized_moves

        processed_df["moves_tokenized"] = processed_df["moves"].apply(tokenize_moves)

        return processed_df
