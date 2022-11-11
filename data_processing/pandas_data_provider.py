import os
from typing import Dict, List

import gin
import pandas as pd

from data_processing.chess_data_generator import ChessDataProvider, ChessDataset
from data_processing.chess_tokenizer import ChessTokenizer
from metric_logging import log_param, log_value
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


# @gin.configurable
class PandasCLLPDataGenerator(ChessDataProvider):
    def __init__(
        self,
        data_path: str = None,
        save_final_df_path: str = None,
        crop_df: int = 3 * 10**6,
        use_one_move=None,
        padding_len=40,
    ):

        print(f"Data path: {data_path}")

        if GlobalParamsHandler().get_data_path() is not None:
            self.data_path = GlobalParamsHandler().get_data_path()
        else:
            self.data_path = data_path

        self.save_final_df_path = save_final_df_path
        self.crop_df = crop_df
        self.use_one_move = use_one_move
        self.padding_len = padding_len

    def create_data(self):
        self.paths_to_data = []
        self.combined_df = pd.DataFrame()
        for dir_data in os.walk(self.data_path):
            print(dir_data)
            for file in dir_data[-1]:
                if ".pkl" in file:
                    self.paths_to_data.append(os.path.join(dir_data[0], file))

        for file in self.paths_to_data:
            print(f"Processing {file}")
            df = pd.read_pickle(file)
            print(f"Read pickle {file}")
            df = df.head(self.crop_df)
            processed_df = self.process_df(df, file)
            self.combined_df = pd.concat([self.combined_df, processed_df], ignore_index=True)
            del df

        if self.save_final_df_path is not None:
            self.combined_df.to_pickle(self.save_final_df_path)

    def process_df(self, df: pd.DataFrame, file: str) -> pd.DataFrame:
        processed_df = df[["input_ids", "labels", "moves"]]
        processed_df.rename(columns={"labels": "subgoal_board", "input_ids": "input_board"}, inplace=True)

        def tokenize_moves(moves: List[str]) -> List[int]:
            tokenized_moves = []
            if self.use_one_move:
                moves = [moves[0]]
            for move in moves:
                tokenized_moves.extend(
                    ChessTokenizer.encode_uci_move(move) + [ChessTokenizer.special_vocab_to_tokens["<SEP>"]]
                )

            tokenized_moves.append(ChessTokenizer.special_vocab_to_tokens["<EOS>"])
            if not self.use_one_move:
                if len(tokenized_moves) < self.padding_len:
                    tokenized_moves.extend(
                        [ChessTokenizer.special_vocab_to_tokens["<PAD>"]] * (self.padding_len - len(tokenized_moves))
                    )

            return tokenized_moves

        data_for_model = {"input_ids": [], "labels": []}

        for idx, row in processed_df.iterrows():
            if len(row["moves"]) > 0:
                data_for_model["input_ids"].append(
                    row["input_board"] + [ChessTokenizer.special_vocab_to_tokens["<SEP>"]] + row["subgoal_board"]
                )
                data_for_model["labels"].append(tokenize_moves(row["moves"]))
                if idx % 1000 == 0:
                    log_value(f"df_processing_{file}", idx, idx / len(processed_df))

        data_df = pd.DataFrame(data_for_model)

        return data_df


class PandasCLLPDataProvider(ChessDataProvider):
    def __init__(self, data_path=None, eval_datapoints: int = 10000):
        if GlobalParamsHandler().get_data_path() is not None:
            data_path = GlobalParamsHandler().get_data_path()
            print(f"Data path: {data_path}")

        print(f"Reading pickle")
        df = pd.read_pickle(data_path)
        print(f"Finished reading pickle")
        print(f"Processing df")
        processed_df = self.process_df(df)
        print(f"Finished processing df")
        self.data_train = self.pandas_to_dict(processed_df.head(-eval_datapoints))
        self.data_eval = self.pandas_to_dict(processed_df.tail(eval_datapoints))

        log_param("Train set size", len(self.data_train))
        log_param("Eval set size", len(self.data_eval))

    def pandas_to_dict(self, df: pd.DataFrame) -> Dict:
        return df.to_dict(orient="records")

    def process_df(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[["input_ids", "labels"]]

    def get_train_set_generator(self) -> ChessDataset:
        return ChessDataset(self.data_train)

    def get_eval_set_generator(self) -> ChessDataset:
        return ChessDataset(self.data_eval)


class PandasBertForSequenceDataProvider(ChessDataProvider):
    def __init__(self, data_path=None, eval_datapoints: int = 10000):
        # if GlobalParamsHandler().get_data_path() is not None:
        #     data_path = GlobalParamsHandler().get_data_path()
        #     print(f"Data path: {data_path}")

        print(f"Reading pickle")
        df = pd.read_pickle(data_path)
        print(f"Finished reading pickle")
        print(f"Processing df")
        processed_df = self.process_df(df)
        print(f"Finished processing df")
        self.data_train = self.pandas_to_dict(processed_df.head(-eval_datapoints))
        self.data_eval = self.pandas_to_dict(processed_df.tail(eval_datapoints))

        log_param("Train set size", len(self.data_train))
        log_param("Eval set size", len(self.data_eval))

    def pandas_to_dict(self, df: pd.DataFrame) -> Dict:
        data = {}
        for (id, (_, row)) in enumerate(df[["input_ids", "Q"]].iterrows()):
            data_for_model = {"input_ids": None, "labels": None}
            data_for_model["input_ids"] = row["input_ids"]
            data_for_model["labels"] = row["Q"]
            data[id] = data_for_model
        return data

    def process_df(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[["input_ids", "Q"]]

    def get_train_set_generator(self) -> ChessDataset:
        return ChessDataset(self.data_train)

    def get_eval_set_generator(self) -> ChessDataset:
        return ChessDataset(self.data_eval)
