#!/home/martyna/miniconda3/envs/subgoal_search_chess/bin/python -u

# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "0"

import os
import sys

sys.path.append("/")
os.environ["SUBGOAL_PROJECT_ROOT"] = "/home/martyna/RESEARCH/CODE/subgoal_search_chess"

from chess_engines.bots.basic_chess_engines import RandomChessEngine
from chess_engines.banksia_gui_uci.banksia_gui_core import main_uci_loop


engine = RandomChessEngine()

if __name__ == "__main__":
    main_uci_loop(engine)
