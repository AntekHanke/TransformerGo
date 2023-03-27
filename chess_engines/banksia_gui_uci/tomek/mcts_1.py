#!/home/tomasz/anaconda3/envs/subgoal_search_chess/bin/python -u

# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "0"

import os
import sys

sys.path.append("/home/tomasz/Research/subgoal_search_chess")
os.environ["SUBGOAL_PROJECT_ROOT"] = "/home/tomasz/Research/subgoal_search_chess"

from chess_engines.banksia_gui_uci.banksia_gui_core import main_uci_loop
from chess_engines.bots.mcts_bot import MCTSChessEngine, SubgoalMCTSChessEngine

GEN_LONG_K_3 = "/home/tomasz/Research/subgoal_chess_data/local_leela_models/long_training/checkpoint-221500"
CLLP_PATH = "/home/tomasz/Research/subgoal_chess_data/local_leela_models/cllp/medium"

engine = SubgoalMCTSChessEngine(
    time_limit=120,
    max_mcts_passes=2,
    generator_path=GEN_LONG_K_3,
    cllp_path=CLLP_PATH,
    cllp_num_beams=8,
    cllp_num_return_sequences=4,
    generator_num_beams=16,
    generator_num_subgoals=8,
    sort_subgoals_by="highest_min_probability",
    num_top_subgoals=4,
)

if __name__ == "__main__":
    main_uci_loop(engine)
