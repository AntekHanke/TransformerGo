#!/home/tomasz/anaconda3/envs/subgoal_search_chess/bin/python -u

# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "0"

import os
import sys
from datetime import date

sys.path.append("/home/tomasz/Research/subgoal_search_chess")
os.environ["SUBGOAL_PROJECT_ROOT"] = "/home/tomasz/Research/subgoal_search_chess"

from chess_engines.banksia_gui_uci.banksia_gui_core import main_uci_loop
from chess_engines.bots.mcts_bot import MCTSChessEngine, SubgoalMCTSChessEngine

GEN_LONG_K_3_b = "/home/tomasz/Research/subgoal_chess_data/local_models/generator_k_3_b"
CLLP_PATH = "/home/tomasz/Research/subgoal_chess_data/local_models/cllp"
STOCKFISH_PATH = None

engine = SubgoalMCTSChessEngine(
    time_limit=120,
    max_mcts_passes=5,
    generator_path=GEN_LONG_K_3_b,
    cllp_path=CLLP_PATH,
    cllp_num_beams=2,
    cllp_num_return_sequences=1,
    generator_num_beams=12,
    generator_num_subgoals=6,
    generator_num_subgoals_first_layer=8,
    sort_subgoals_by="highest_total_probability",
    num_top_subgoals=6,
    log_trees= True,
    log_dir = f"/home/tomasz/Research/subgoal_chess_data/mcts_trees",
)

if __name__ == "__main__":
    main_uci_loop(engine)
