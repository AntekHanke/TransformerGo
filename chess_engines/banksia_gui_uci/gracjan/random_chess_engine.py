#!/home/gracjan/anaconda3/envs/subgoal_chess/bin/python3 -u
import sys
import os
sys.path.append("/home/gracjan/subgoal/subgoal_search_chess-uci_engines")
os.environ["SUBGOAL_PROJECT_ROOT"] = "/home/gracjan/subgoal/subgoal_search_chess-uci_engines"

from chess_engines.banksia_gui_uci.gracjan.banksia_gui_core import main_uci_loop
from chess_engines.bots.basic_chess_engines import RandomChessEngine

engine = RandomChessEngine()

if __name__ == "__main__":
    main_uci_loop(engine)
