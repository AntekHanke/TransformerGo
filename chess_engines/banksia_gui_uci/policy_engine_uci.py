#!/home/tomasz/anaconda3/envs/subgoal_search_chess/bin/python -u

from set_root_path import set_root_path

set_root_path()

from chess_engines.banksia_gui_uci.banksia_gui_core import main_uci_loop
from chess_engines.bots.basic_chess_engines import PolicyChess

engine = PolicyChess("/home/gracjan/subgoal/subgoal_chess_data/policy/final_model", "/home/gracjan/subgoal/log_exp_subgoal_chess_engines", debug_mode=True)

if __name__ == "__main__":
    main_uci_loop(engine)
