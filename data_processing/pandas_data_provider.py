import random
import time
from itertools import cycle

from configures.global_config import MAX_MOVES_FOR_CLLP
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


class IterableDataLoader(IterableDataset):
    def __init__(
        self,
        data_path: str,
        files_batch_size: int = 10,
        p_sample: Optional[float] = None,
        cycle: bool = True,
        database: str = "leela",
    ) -> None:
        self.data_path = data_path
        self.files_batch_size = files_batch_size
        self.p_sample = p_sample
        self.cycle = cycle
        self.database = database

        self.files_names: List[str] = []
        self.prepare_files_list()

    def prepare_files_list(self):
        if os.path.isfile(self.data_path):
            self.files_names.append(self.data_path)
        else:
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
    def process_df(df: pd.DataFrame) -> List[Dict[str, List[int]]]:
        raise NotImplementedError

    def generate_data(self) -> Iterator[Dict[str, List[int]]]:

        data: List[Dict[str, List[int]]] = []

        if self.cycle:
            files_iterable = enumerate(cycle(self.files_names))
        else:
            files_iterable = enumerate(self.files_names)

        for file_num, path_to_file in files_iterable:
            load_df: pd.DataFrame = pd.read_pickle(path_to_file)
            log_value("load_df", file_num, file_num)

            if self.p_sample:
                log_value("p_sample", 0, self.p_sample)
                load_df = load_df.sample(frac=self.p_sample, random_state=1)

            data.extend(self.process_df(load_df))
            if (file_num + 1) % self.files_batch_size == 0 or (file_num + 1) == len(self.files_names):
                print(f"Will shuflle data from {file_num - self.files_batch_size + 1} to {file_num} len = {len(data)}")
                time_s = time.time()
                random.shuffle(data)
                print(f"Shuffled in {time.time() - time_s}")
                for x in data:
                    yield x
                data.clear()

    def log_samples(self, log_samples_limit: int):
        raise NotImplementedError


class IterableSubgoalDataLoader(IterableDataLoader):
    @staticmethod
    def process_df(df: pd.DataFrame) -> pd.DataFrame:
        df = df[["input_ids", "labels"]]
        return df.to_dict(orient="records")

    def __getitem__(self, item):
        raise NotImplementedError

    def log_samples(self, log_samples_limit: int):
        raise NotImplementedError


class IterablePolicyDataLoader(IterableDataLoader):
    @staticmethod
    def process_df(df: pd.DataFrame) -> pd.DataFrame:
        df = df[["input_ids", "labels"]]
        return df.to_dict(orient="records")

    def __getitem__(self, item):
        raise NotImplementedError

    def log_samples(self, log_samples_limit: int):
        raise NotImplementedError


class IterableSubgoalToPolicyDataLoader(IterableDataLoader):
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

    def __getitem__(self, item):
        raise NotImplementedError

    def log_samples(self, log_samples_limit: int):
        raise NotImplementedError


class IterableCLLPDataLoader(IterableDataLoader):
    def process_df(self, df: pd.DataFrame):
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

    def __getitem__(self, item):
        raise NotImplementedError

    def log_samples(self, log_samples_limit: int):
        raise NotImplementedError


class PandasBertForSequenceDataProvider(ChessDataProvider):
    def __init__(self, data_path=None, eval_datapoints: int = 10000):

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
        for (id, (_, row)) in enumerate(df[["target_immutable_board", "Q"]].iterrows()):
            data[id] = {
                "input_ids": ChessTokenizer.encode_immutable_board(row["target_immutable_board"]),
                "labels": row["Q"],
            }
        return data

    def process_df(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[["target_immutable_board", "Q"]]

    def get_train_set_generator(self) -> ChessDataset:
        return ChessDataset(self.data_train)

    def get_eval_set_generator(self) -> ChessDataset:
        return ChessDataset(self.data_eval)
