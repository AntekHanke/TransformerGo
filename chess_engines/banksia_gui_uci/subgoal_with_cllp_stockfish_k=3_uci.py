#!/home/gracjan/anaconda3/envs/subgoal_chess/bin/python -u

from set_root_path import set_root_path

set_root_path()

from chess_engines.banksia_gui_uci.banksia_gui_core import main_uci_loop
from chess_engines.bots.basic_chess_engines import SubgoalWithCLLPStockfish

path_to_generator: str = "/home/gracjan/subgoal/subgoal_chess_data/generator/final_model"
path_to_cllp: str = "/home/gracjan/subgoal/subgoal_chess_data/cllp/final_model"
log_dir: str = "/home/gracjan/subgoal/log_exp_subgoal_chess_engines"

engine = SubgoalWithCLLPStockfish(
    generator_checkpoint=path_to_generator,
    cllp_checkpoint=path_to_cllp,
    n_subgoals=3,
    stockfish_depth=20,
    log_dir=log_dir,
    debug_mode=True,
    replace_legall_move_with_random=False
)

if __name__ == "__main__":
    main_uci_loop(engine)
