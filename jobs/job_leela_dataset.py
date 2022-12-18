import os


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
