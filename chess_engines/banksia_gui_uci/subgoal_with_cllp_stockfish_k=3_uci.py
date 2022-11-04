#!/home/tomasz/anaconda3/envs/subgoal_search_chess/bin/python -u

from set_root_path import set_root_path

set_root_path()

from chess_engines.banksia_gui_uci.banksia_gui_core import main_uci_loop
from chess_engines.bots.basic_chess_engines import SubgoalWithCLLPStockfish

path_to_generator: str = ""
path_to_cllp: str = ""
log_dir: str = ""

engine = SubgoalWithCLLPStockfish(
    generator_checkpoint=path_to_generator,
    cllp_checkpoint=path_to_cllp,
    n_subgoals=3,
    stockfish_depth=20,
    log_dir="/home/tomasz/Research/subgoal_chess_data/bot_logs",
)

if __name__ == "__main__":
    main_uci_loop(engine)
