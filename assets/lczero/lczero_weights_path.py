import os

from utils.detect_local_machine import is_local_machine, get_local_machine

if is_local_machine():
    if get_local_machine() == "tomasz":
        DEFAULT_LC_ZERO_WEIGHTS_PATH = "/home/tomasz/Research/subgoal_chess_data/lczero/medium"
    elif get_local_machine() == "gracjan":
        DEFAULT_LC_ZERO_WEIGHTS_PATH = None
else:
    DEFAULT_LC_ZERO_WEIGHTS_PATH = "/main_storage/lczero/medium"
