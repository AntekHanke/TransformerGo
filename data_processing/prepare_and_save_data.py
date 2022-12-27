import os
import random
import time
from pathlib import Path
from typing import List, Optional

import pandas as pd
from tqdm import tqdm

from configures.global_config import MAX_MOVES_FOR_CLLP
from data_processing.chess_tokenizer import ChessTokenizer
from metric_logging import log_object, log_value


class PandasPrepareAndSaveData:
    def __init__(
        self,
        data_path: str = None,
        out_path: str = None,
        files_batch_size: int = 10,
        p_sample: Optional[float] = None,
        create_files_list: bool = True,
    ) -> None:
        self.data_path = data_path
        self.out_path = out_path
        self.files_batch_size = files_batch_size
        self.p_sample = p_sample
        self.eval = eval

        self.files_names: List[str] = []
        if create_files_list:
            self.prepare_files_list()

        self.file_names_queue = {i: [] for i in range(1, MAX_MOVES_FOR_CLLP + 1)}

    def prepare_files_list(self):
        for folder_name in tqdm(os.listdir(self.data_path)):
            path: str = self.data_path + "/" + str(folder_name)
            for file_name in os.listdir(path):
                path_to_file: str = os.path.join(path, file_name)
                if os.path.isfile(path_to_file):
                    self.files_names.append(path_to_file)

        random.shuffle(self.files_names)

    def generate_data(self):
        data = []
        saved_files = 0
        total_data_collected = 0

        for file_num, path_to_file in enumerate(self.files_names):

            log_object("path_to_file", path_to_file)
            log_value("file_num", file_num, file_num / len(self.files_names))

            print(f"Loading {path_to_file}")
            load_df: pd.DataFrame = pd.read_pickle(path_to_file)
            if self.p_sample is not None:
                load_df = load_df.sample(frac=self.p_sample, random_state=1)

            log_value("len_of_df", file_num, len(load_df))

            data.extend(self.process_df(load_df))
            print(f"Data len after processing: {len(data)}")
            if (file_num + 1) % self.files_batch_size == 0 or (file_num + 1) == len(self.files_names):
                time_s = time.time()
                random.shuffle(data)
                log_value("shuffle_time", file_num, time.time() - time_s)
                print(f"Shuffled in {time.time() - time_s}")

                total_data_collected += len(data)
                log_value("total_data_collected", file_num, total_data_collected)

                df = pd.DataFrame(data)
                new_output_dir = Path(f"{self.out_path}/part_{file_num}")
                new_output_dir.mkdir(parents=True, exist_ok=True)
                df.to_pickle(f"{self.out_path}/part_{file_num}/data_{file_num}.pkl")
                saved_files += 1
                log_value("saved_files", saved_files, saved_files)
                data.clear()

    def process_df(self, df: pd.DataFrame):
        raise NotImplementedError


class PandasPolicyPrepareAndSaveData(PandasPrepareAndSaveData):
    def process_df(self, df: pd.DataFrame):
        df = df[["input_idx", "input_ids", "moves", "P"]]
        # data_list = df.to_dict(orient="records")

        current_input_idx = None
        current_best_p = 0
        local_group_of_datapoints = []

        # data = []
        local_groups = []
        local_group_len = []

        for row in df.itertuples():
            if row.input_idx == current_input_idx:
                local_group_of_datapoints.append(row)
            else:
                local_groups.append(local_group_of_datapoints)
                local_group_len.append(len(local_group_of_datapoints))
                local_group_of_datapoints = []
                local_group_of_datapoints.append(row)
            current_input_idx = row.input_idx

        def process_single_local_group(local_group):
            sorted_moves = sorted(local_group, key=lambda x: x.P, reverse=True)
            if len(sorted_moves) == 0:
                return None
            best_move = sorted_moves[0]
            if len(best_move.moves) == 0:
                return None
            return {
                "input_ids": best_move.input_ids + [ChessTokenizer.vocab_to_tokens["<SEP>"]],
                "labels": ChessTokenizer.encode(best_move.moves[0]),
            }

        return [
            process_single_local_group(local_group)
            for local_group in local_groups
            if process_single_local_group(local_group) is not None
        ]


class CLLPPrepareAndSaveData(PandasPrepareAndSaveData):
    def prepare_files_list_k(self):
        for k in range(1, MAX_MOVES_FOR_CLLP + 1):
            for folder_name in tqdm(os.listdir(self.data_path + f"/subgoal_{k}")):
                path: str = self.data_path + f"/subgoal_{k}" + "/" + str(folder_name)
                for file_name in os.listdir(path):
                    path_to_file: str = os.path.join(path, file_name)
                    if os.path.isfile(path_to_file):
                        self.file_names_queue[k].append(path_to_file)

        for k in range(1, MAX_MOVES_FOR_CLLP + 1):
            random.shuffle(self.file_names_queue[k])

        for _ in range(2):
            self.files_names += [self.file_names_queue[k].pop() for k in range(1, MAX_MOVES_FOR_CLLP + 1)]

        d = 5


    def process_df(self, df: pd.DataFrame):
        df = df[["input_ids", "labels", "moves"]]
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

        return [process_single_datapoint(datapoint) for datapoint in data_list if len(datapoint["moves"]) > 0]
