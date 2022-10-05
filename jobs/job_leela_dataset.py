from jobs.core import Job
from typing import List
from leela.leela_data_creator import LeelaTressGenerator


class LeelaDatasetGenerator(Job):
    def __init__(
            self,
            path_to_chess_dataset: str,
            leela_parms: List[str],
            number_of_searching_nodes: int = 1000
    ) -> None:
        self.path_to_chess_dataset = path_to_chess_dataset
        self.leela_parms = leela_parms
        self.number_of_searching_nodes = number_of_searching_nodes

    def execute(self):
        leela_gegenartor = LeelaTressGenerator(
            self.path_to_chess_dataset,
            self.leela_parms,
            self.number_of_searching_nodes
        )
        leela_gegenartor.generate_and_save_tress_by_leela()
