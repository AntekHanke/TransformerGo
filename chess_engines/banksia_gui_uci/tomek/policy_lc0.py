#!/home/tomasz/anaconda3/envs/subgoal_search_chess/bin/python -u
import os
import sys

sys.path.append("/home/tomasz/Research/subgoal_search_chess")
os.environ["SUBGOAL_PROJECT_ROOT"] = "/home/tomasz/Research/subgoal_search_chess"

from chess_engines.bots.basic_chess_engines import PolicyChess
from chess_engines.banksia_gui_uci.gracjan.banksia_gui_core import main_uci_loop


log_dir: str = "/home/tomasz/Research/subgoal_chess_data/bot_logs"


engine = PolicyChess(
    policy_checkpoint=None,
    log_dir=log_dir,
    debug_mode=True,
    replace_legall_move_with_random=False,
    do_sample=False,
    name="LeelaChessZero_POLICY",
    use_lczero_policy=True,
)

if __name__ == "__main__":
    main_uci_loop(engine)
