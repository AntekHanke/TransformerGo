from typing import Type

from data_processing.auxiliary_code_for_data_processing.pgn.prepare_and_save_data import PandasPrepareAndSaveData
from jobs.core import Job


class LeelaDatasetGenerator(Job):
    def __init__(self, mcts_gen_class):
        self.mcts_gen = mcts_gen_class()

    def execute(self):
        self.mcts_gen.generate_data()


class LeelaParallelDatasetGenerator(Job):
    def __init__(self, mcts_gen_class):
        self.mcts_gen = mcts_gen_class()

    def execute(self):
        self.mcts_gen.generate_parallel_data_from_path()


# @gin.configurable
class LeelaCCLPDataProcessing(Job):
    def __init__(self, pandas_data_provider_class=None):
        self.pandas_data_provider = pandas_data_provider_class()

    def execute(self):
        self.pandas_data_provider.create_data()


class LeelaPrepareAndSaveData(Job):
    def __init__(self, pandas_data_prepare_cls: Type[PandasPrepareAndSaveData]):
        self.gen = pandas_data_prepare_cls()

    def execute(self):
        self.gen.generate_data()
