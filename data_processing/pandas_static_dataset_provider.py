import glob
import os
import random
from typing import Dict, Optional, List

import pandas as pd
import torch

from metric_logging import log_object, log_value


# class ChessDataset(torch.utils.data.Dataset):
#     """Used by Pytorch DataLoader to get batches of data."""
#
#     def __init__(self, data: Dict):
#         self.data = data
#
#     def __getitem__(self, idx: int):
#         return self.data[idx]
#
#     def __len__(self):
#         return len(self.data)

class PandasStaticDataProvider(torch.utils.data.Dataset):
    def __init__(
        self,
        data_path: str,
        files_batch_size: int = 10,
        p_sample: Optional[float] = None,
        name: str = "default",
    ) -> None:

        self.data_path = data_path
        self.files_batch_size = files_batch_size
        self.p_sample = p_sample
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
        random.shuffle(self.files_names)
        log_object(f"files_names_after_shuffle", self.files_names)

        assert len(self.files_names) > 0, f"No data files found in {self.data_path}"


    def __getitem__(self, idx: int):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

    # @staticmethod
    # def process_df(df: pd.DataFrame) -> List[Dict[str, List[int]]]:
    #     raise NotImplementedError

    @staticmethod
    def process_df(df: pd.DataFrame):
        df = df[["input_ids", "labels"]]
        return df.to_dict(orient="records")

    def generate_data(self):

        if self.p_sample:
            log_value(f"p_sample_{self.name}", 0, self.p_sample)

        data: List[Dict[str, List[int]]] = []

        for file_num, path_to_file in self.files_names:
            load_df: pd.DataFrame = pd.read_pickle(path_to_file)

            if len(load_df) == 0:
                log_object("Empty file", path_to_file)
                continue

            self.successfully_loaded_files += 1

            log_value(f"load_df_all_files_{self.name}", self.successfully_loaded_files, self.successfully_loaded_files)

            if self.p_sample:
                load_df = load_df.sample(frac=self.p_sample)

            data.extend(self.process_df(load_df))

