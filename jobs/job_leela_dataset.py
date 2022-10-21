import os

from data_processing.mcts_data_generator import SubgoalMCGamesDataGenerator
from jobs.core import Job

class LeelaDatasetGenerator(Job):
    def __init__(self, mcts_gen_class):
        self.mcts_gen = mcts_gen_class()

    def execute(self):
        self.mcts_gen.generate_data()
