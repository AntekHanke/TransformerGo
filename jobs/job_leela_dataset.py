from data_processing.mcts_data_generator import SubgoalMCGamesDataGenerator
from jobs.core import Job
from typing import List
from leela.leela_data_creator import LeelaTressGenerator


class LeelaDatasetGenerator(Job):
    def __init__(self):
        self.generator = SubgoalMCGamesDataGenerator()

    def execute(self):
        leela_gegenartor = LeelaTressGenerator(
            self.path_to_chess_dataset,
            self.leela_parms,
            self.number_of_searching_nodes
        )
        leela_gegenartor.generate_and_save_tress_by_leela()
