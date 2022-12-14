#!/home/tomasz/anaconda3/envs/subgoal_search_chess/bin/python -u

from set_root_path import set_root_path

set_root_path()

from chess_engines.banksia_gui_uci.banksia_gui_core import main_uci_loop
from chess_engines.bots.basic_chess_engines import SubgoalWithCLLP

engine = SubgoalWithCLLP(
    "/home/tomasz/Research/subgoal_chess_data/local_leela_models/generator/medium_k=1/final_model",
    "/home/tomasz/Research/subgoal_chess_data/local_leela_models/cllp_all_moves/final_model",
)

if __name__ == "__main__":
    main_uci_loop(engine)
