#!/home/tomasz/anaconda3/envs/subgoal_search_chess/bin/python -u

from set_root_path import set_root_path

set_root_path()

from chess_engines.banksia_gui_uci.banksia_gui_core import main_uci_loop
from chess_engines.bots.basic_chess_engines import SubgoalWithCLLPStockfish

path_to_generator: str = "/home/tomasz/Research/subgoal_chess_data/local_leela_models/generator/medium_k=3/final_model"
path_to_cllp: str = "/home/tomasz/Research/subgoal_chess_data/local_leela_models/cllp_all_moves/final_model"
log_dir: str = "/home/tomasz/Research/subgoal_chess_data/bot_logs"

engine = SubgoalWithCLLPStockfish(
    name="subgoal_with_cllp_stockfish_k=3",
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
