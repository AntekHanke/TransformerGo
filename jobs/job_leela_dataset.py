import os

from data_processing.mcts_data_generator import SubgoalMCGamesDataGenerator
from jobs.core import Job

class LeelaDatasetGenerator(Job):
    def __init__(self, mcts_gen_class):
        self.mcts_gen = mcts_gen_class()

    def execute(self):
        # with open("/tmp/lustre/plggracjangoral/leela_chess/mrunner_scratch/subgoal-chess/05_10-14_53-objective_feynman/leela-dataset-part-1_35l1_146/leela/trees_of_leela/all_trees.txt", "r") as f:
        #     lines =  f.readlines()
        #     for line in lines:
        #         print(line)
        #         break
        for x in os.walk("/tmp/lustre/plggracjangoral/leela_chess/mrunner_scratch/subgoal-chess/05_10-14_53-objective_feynman"):
            print(x)
        self.mcts_gen.generate_data()
