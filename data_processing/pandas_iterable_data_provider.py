import glob
import os
import random
import time
from itertools import cycle
from typing import List, Dict, Iterator, Optional

import pandas as pd
from torch.utils.data import IterableDataset

from data_processing.chess_data_generator import ChessDataProvider, ChessDataset
from data_processing.chess_tokenizer import ChessTokenizer
from data_processing.data_processing_functions import (
    subgoal_process_df,
    policy_process_df,
    policy_with_history_process_df,
    subgoal_to_policy_process_df,
    cllp_process_df,
)
from metric_logging import log_param, log_value, log_object


class PandasIterableDataProvider(IterableDataset):
    def __init__(
        self,
        data_path: str,
        files_batch_size: int = 10,
        p_sample: Optional[float] = None,
        cycle: bool = True,
        name: str = "default",
    ) -> None:

        self.data_path = data_path
        self.files_batch_size = files_batch_size
        self.p_sample = p_sample
        self.cycle = cycle
        self.name = name

        self.successfully_loaded_files = 0
        self.files_names: List[str] = []
        self.prepare_files_list()

    def prepare_files_list(self):
        if os.path.isfile(self.data_path):
            self.files_names.append(self.data_path)
        else:
            if self.data_path.endswith("/"):
                self.data_path = self.data_path[:-1]
            self.files_names = list(glob.glob(f"{self.data_path}/**/*.pkl", recursive=True))

        log_object(f"{self.name}_files_names_before_shuffle", self.files_names)
        random.Random(time.time()).shuffle(self.files_names)
        log_object(f"files_names_after_shuffle", self.files_names)

        assert len(self.files_names) > 0, f"No data files found in {self.data_path}"

    def __iter__(self) -> Iterator[Dict[str, List[int]]]:
        return self.generate_data()

    def __getitem__(self, item):
        raise NotImplementedError

    @staticmethod
    def process_df(df: pd.DataFrame) -> List[Dict[str, List[int]]]:
        raise NotImplementedError

    def generate_data(self) -> Iterator[Dict[str, List[int]]]:

        if self.p_sample:
            log_value("p_sample", 0, self.p_sample)

        data: List[Dict[str, List[int]]] = []

        if self.cycle:
            files_iterable = enumerate(cycle(self.files_names))
        else:
            files_iterable = enumerate(self.files_names)

        for file_num, path_to_file in files_iterable:
            load_df: pd.DataFrame = pd.read_pickle(path_to_file)

            if len(load_df) == 0:
                log_object("Empty file", path_to_file)
                continue

            self.successfully_loaded_files += 1
            log_value(f"load_df_all_files_{self.name}", self.successfully_loaded_files, self.successfully_loaded_files)

            if self.p_sample:
                load_df = load_df.sample(frac=self.p_sample)

            data.extend(self.process_df(load_df))
            if (self.successfully_loaded_files + 1) % self.files_batch_size == 0 or (file_num + 1) == len(
                self.files_names
            ):
                random.Random(time.time()).shuffle(data)
                for x in data:
                    yield x
                data.clear()


class PandasIterableSubgoalDataProvider(PandasIterableDataProvider):
    @staticmethod
    def process_df(df: pd.DataFrame):
        return subgoal_process_df(df)


class PandasIterablePolicyDataProvider(PandasIterableDataProvider):
    @staticmethod
    def process_df(df: pd.DataFrame) -> pd.DataFrame:
        return policy_process_df(df)


class PandasIterablePolicyWithHistoryDataProvider(PandasIterableDataProvider):
    @staticmethod
    def process_df(df: pd.DataFrame) -> pd.DataFrame:
        return policy_with_history_process_df(df)


class PandasIterableSubgoalToPolicyDataProvider(PandasIterableDataProvider):
    @staticmethod
    def process_df(df: pd.DataFrame):
        return subgoal_to_policy_process_df(df)


class PandasIterableCLLPDataProvider(PandasIterableDataProvider):
    @staticmethod
    def process_df(df: pd.DataFrame):
        return cllp_process_df(df)


class PandasBertForSequenceDataProvider(ChessDataProvider):
    def __init__(self, data_path=None, eval_datapoints: int = 10000):
        df = pd.read_pickle(data_path)
        processed_df = self.process_df(df)
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
