#!/home/tomasz/anaconda3/envs/subgoal_search_chess/bin/python -u

import os

if "SUBGOAL_PROJECT_ROOT" not in os.environ:
    raise Exception(
        "SUBGOAL_PROJECT_ROOT not in os.environ, please set this variable pointing to the root of the project"
    )
import sys
sys.path.append(os.environ["SUBGOAL_PROJECT_ROOT"])

from chess_engines.banksia_gui_uci.banksia_gui_core import main_uci_loop
from chess_engines.bots.basic_chess_engines import PolicyChess

engine = PolicyChess("/home/tomasz/Research/subgoal_chess_data/local_leela_models/policy/final_model")

if __name__ == "__main__":
    main_uci_loop(engine)
