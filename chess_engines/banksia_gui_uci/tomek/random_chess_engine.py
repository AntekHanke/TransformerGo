#!/home/gracjan/anaconda3/envs/subgoal_chess/bin/python3 -u
from set_root_path import set_root_path

set_root_path()

from chess_engines.banksia_gui_uci.banksia_gui_core import main_uci_loop
from chess_engines.bots.basic_chess_engines import RandomChessEngine

engine = RandomChessEngine()

if __name__ == "__main__":
    main_uci_loop(engine)
