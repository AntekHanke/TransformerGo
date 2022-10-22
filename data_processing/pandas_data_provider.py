from typing import Dict, List

import pandas as pd

from data_processing.chess_data_generator import ChessDataProvider, ChessDataset
from utils.global_params_handler import GlobalParamsHandler


class PandasSubgoalDataProvider(ChessDataProvider):
    def __init__(self, data_path = None, eval_datapoints: int = 10000):
        if data_path is None:
            data_path = GlobalParamsHandler().get_data_path()
            print(f"Data path: {data_path}")

        df = pd.read_pickle(data_path)
        processed_df = self.process_df(df)
        self.data_train = self.pandas_to_dict(processed_df.head(-eval_datapoints))
        self.data_eval = self.pandas_to_dict(processed_df.tail(eval_datapoints))

    def process_df(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[['input_ids', 'labels']]

    def pandas_to_dict(self, df: pd.DataFrame) -> Dict:
        return df.to_dict(orient="records")

    def get_train_set_generator(self) -> ChessDataset:
        return ChessDataset(self.data_train)

    def get_eval_set_generator(self) -> ChessDataset:
        return ChessDataset(self.data_eval)


class PandasCLLPDataProvider(ChessDataProvider):
    def __init__(self, data_path: List[str], eval_datapoints: int = 10000):
        if data_path is None:
            data_path = GlobalParamsHandler().get_data_path()
        self.data_path = data_path
    def process_df(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[['input_ids', 'labels']]
