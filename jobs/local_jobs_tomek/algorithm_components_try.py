from policy.cllp import CLLP
from subgoal_generator.subgoal_generator import BasicChessSubgoalGenerator

subgoal_generator = BasicChessSubgoalGenerator("/home/tomasz/Research/subgoal_chess_data/local_leela_models/4gpu_generator/subgoals_k=3")
cllp = CLLP("/home/tomasz/Research/subgoal_chess_data/local_leela_models/cllp/medium")