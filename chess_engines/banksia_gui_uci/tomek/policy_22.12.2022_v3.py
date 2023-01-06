#!/home/tomasz/anaconda3/envs/subgoal_search_chess/bin/python -u
import os
import sys

sys.path.append("/home/tomasz/Research/subgoal_search_chess")
os.environ["SUBGOAL_PROJECT_ROOT"] = "/home/tomasz/Research/subgoal_search_chess"

from chess_engines.bots.basic_chess_engines import PolicyChess
from chess_engines.banksia_gui_uci.gracjan.banksia_gui_core import main_uci_loop


policy_checkpoint: str = "/home/tomasz/Research/subgoal_chess_data/local_leela_models/policy/23.12.2022_medium.03"
log_dir: str = "/home/tomasz/Research/subgoal_chess_data/bot_logs"


engine = PolicyChess(
    policy_checkpoint=policy_checkpoint,
    log_dir=log_dir,
    debug_mode=True,
    replace_legall_move_with_random=False,
    name="medium_3",
)

if __name__ == "__main__":
    main_uci_loop(engine)
