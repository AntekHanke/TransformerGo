import random
from data_processing.chess_data_generator import ChessDataProvider, ChessDataset
from data_processing.chess_tokenizer import ChessTokenizer
from data_processing.data_utils import immutable_boards_to_img
from metric_logging import log_param, log_value, log_object
import os
from os.path import isfile, join
from typing import List, Dict, Iterator, Optional, Union
import pandas as pd
from tqdm import tqdm
from torch.utils.data import IterableDataset
from utils.global_params_handler import GlobalParamsHandler

# rng = random.Random(0)

# class PandasSubgoalDataProvider(ChessDataProvider):
#     """
#     This class supports CLLP and Genarator model
#     """
#
#     def __init__(
#         self, data_path: Optional[str] = None, eval_datapoints: int = 100000, random_half: bool = False
#     ) -> None:
#
#         df: pd.DataFrame = pd.DataFrame()
#
#         if data_path is None:
#             data_path = GlobalParamsHandler().get_data_path()
#             print(f"Data path: {data_path}")
#
#         for folder_name in tqdm(os.listdir(data_path)):
#             path: str = data_path + "/" + str(folder_name)
#             name_of_files: List[str] = [f for f in os.listdir(path) if isfile(join(path, f))]
#             for name_of_file in tqdm(name_of_files):
#                 load_df: pd.DataFrame = pd.read_pickle(path + "/" + name_of_file)
#
#                 if random_half:
#                     load_df = load_df.sample(frac=0.5, random_state=1)
#
#                 df: pd.DataFrame = pd.concat([df, load_df], ignore_index=True)
#
#         processed_df: pd.DataFrame = self.process_df(df)
#         self.data_train: Dict = self.pandas_to_dict(processed_df.head(-eval_datapoints))
#         self.data_eval: Dict = self.pandas_to_dict(processed_df.tail(eval_datapoints))
#
#         log_param("Train set size", len(self.data_train))
#         log_param("Eval set size", len(self.data_eval))
#
#     def get_train_set_generator(self) -> ChessDataset:
#         return ChessDataset(self.data_train)
#
#     def get_eval_set_generator(self) -> ChessDataset:
#         return ChessDataset(self.data_eval)
#
#     @staticmethod
#     def process_df(df: pd.DataFrame) -> pd.DataFrame:
#         return df[["input_ids", "labels"]]
#
#     @staticmethod
#     def pandas_to_dict(df: pd.DataFrame) -> Dict:
#         return df.to_dict(orient="records")
#
#
# # @gin.configurable
# class PandasCLLPDataGenerator:
#     def __init__(
#         self,
#         data_path: Optional[str] = None,
#         save_final_df_path: Optional[str] = None,
#         use_one_move=None,
#         padding_len: int = 40,
#     ) -> None:
#         print(f"Data path: {data_path}")
#
#         if GlobalParamsHandler().get_data_path() is not None:
#             self.data_path = GlobalParamsHandler().get_data_path()
#         else:
#             self.data_path = data_path
#
#         self.save_final_df_path = save_final_df_path
#         self.use_one_move = use_one_move
#         self.padding_len = padding_len
#         self.paths_to_data: Optional[List[str]] = None
#
#     def create_data(self):
#         self.paths_to_data = []
#         for dir_data in os.walk(self.data_path):
#             print(dir_data)
#             for file in dir_data[-1]:
#                 if ".pkl" in file:
#                     self.paths_to_data.append(os.path.join(dir_data[0], file))
#
#         for file in tqdm(self.paths_to_data):
#             print(f"Processing {file}")
#             df: pd.DataFrame = pd.read_pickle(file)
#             print(f"Read pickle {file}")
#             df = self.process_df(df, file)
#
#             if self.save_final_df_path is not None:
#                 path_elements: List[str] = file.split("/")
#                 df.to_pickle(self.save_final_df_path + "/" + "cllp_" + path_elements[-1])
#
#     def process_df(self, df: pd.DataFrame, file: str) -> pd.DataFrame:
#         processed_df = df[["input_ids", "labels", "moves"]]
#         processed_df = processed_df.rename(columns={"labels": "subgoal_board", "input_ids": "input_board"})
#
#         def tokenize_moves(moves: List[str]) -> List[int]:
#             tokenized_moves = []
#             if self.use_one_move:
#                 moves = [moves[0]]
#             for move in moves:
#                 tokenized_moves.extend(
#                     ChessTokenizer.encode_uci_move(move) + [ChessTokenizer.special_vocab_to_tokens["<SEP>"]]
#                 )
#
#             tokenized_moves.append(ChessTokenizer.special_vocab_to_tokens["<EOS>"])
#             if not self.use_one_move:
#                 if len(tokenized_moves) < self.padding_len:
#                     tokenized_moves.extend(
#                         [ChessTokenizer.special_vocab_to_tokens["<PAD>"]] * (self.padding_len - len(tokenized_moves))
#                     )
#
#             return tokenized_moves
#
#         data_for_model = {"input_ids": [], "labels": []}
#
#         for idx, row in processed_df.iterrows():
#             if len(row["moves"]) > 0:
#                 data_for_model["input_ids"].append(
#                     row["input_board"] + [ChessTokenizer.special_vocab_to_tokens["<SEP>"]] + row["subgoal_board"]
#                 )
#                 data_for_model["labels"].append(tokenize_moves(row["moves"]))
#                 if idx % 1000 == 0:
#                     log_value(f"df_processing_{file}", idx, idx / len(processed_df))
#
#         data_df = pd.DataFrame(data_for_model)
#
#         return data_df
#


class IterableDataLoader(IterableDataset):
    def __init__(
        self,
        data_path: Union[str, List[str]],
        files_batch_size: int = 10,
        take_random_half_of_data: bool = False,
        # log_samples_limit: Optional[int] = None,
    ) -> None:
        self.data_path = data_path
        self.files_batch_size = files_batch_size
        self.take_random_half_of_data = take_random_half_of_data
        self.eval = eval

        self.files_names: List[str] = []
        self.prepare_files_list()

    def prepare_files_list(self):

        for folder_name in tqdm(os.listdir(self.data_path)):
            path: str = self.data_path + "/" + str(folder_name)
            for file_name in os.listdir(path):
                path_to_file: str = join(path, file_name)
                if isfile(path_to_file):
                    self.files_names.append(path_to_file)

    def __iter__(self) -> Iterator[Dict[str, List[int]]]:
        return self.generate_data()

    def __getitem__(self, item):
        raise NotImplementedError

    @staticmethod
    def process_df(df: pd.DataFrame) -> List[Dict[str, int]]:
        raise NotImplementedError

    def generate_data(self) -> Iterator[Dict[str, List[int]]]:
        data: List[Dict[str, List[int]]] = []
        for file_num, path_to_file in enumerate(self.files_names):
            load_df: pd.DataFrame = pd.read_pickle(path_to_file)

            if self.take_random_half_of_data:
                load_df = load_df.sample(frac=0.5, random_state=1)

            data.extend(self.process_df(load_df))
            if file_num + 1 % self.files_batch_size == 0 or file_num + 1 == len(self.files_names):
                for x in data:
                    yield x

    def log_samples(self, log_samples_limit: int):
        raise NotImplementedError


class IterableSubgoalDataLoader(IterableDataLoader):
    @staticmethod
    def process_df(df: pd.DataFrame) -> pd.DataFrame:
        df = df[["input_ids", "labels"]]
        return df.to_dict(orient="records")


class IterablePolicyDataLoader(IterableDataLoader):
    def process_df(self, df: pd.DataFrame):

        df = df[["input_ids", "moves"]]
        data_list = df.to_dict(orient="records")

        def process_single_datapoint(datapoint):
            return {
                "input_ids": datapoint["input_ids"] + [ChessTokenizer.vocab_to_tokens["<SEP>"]],
                "labels": ChessTokenizer.encode(datapoint["moves"][0]),
            }

        data = [process_single_datapoint(datapoint) for datapoint in data_list if len(datapoint["moves"]) > 0]
        return data
