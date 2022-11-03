#!/home/tomasz/anaconda3/envs/subgoal_search_chess/bin/python -u

from set_root_path import set_root_path

set_root_path()

from chess_engines.banksia_gui_uci.banksia_gui_core import main_uci_loop
from chess_engines.bots.basic_chess_engines import SubgoalWithCLLPStockfish

engine = SubgoalWithCLLPStockfish(
    generator_checkpoint="/home/tomasz/Research/subgoal_chess_data/local_leela_models/generator/medium_k=3/final_model",
    cllp_checkpoint="/home/tomasz/Research/subgoal_chess_data/local_leela_models/cllp_all_moves/final_model",
    n_subgoals=3,
    stockfish_depth=20,
    log_dir="/home/tomasz/Research/subgoal_chess_data/bot_logs",
)

if __name__ == "__main__":
    main_uci_loop(engine)
