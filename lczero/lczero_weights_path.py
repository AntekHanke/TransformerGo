import os

from utils.detect_local_machine import is_local_machine, get_local_machine

if is_local_machine():
    if get_local_machine() == "tomasz":
        DEFAULT_LC_ZERO_WEIGHTS_PATH = "/home/tomasz/Research/subgoal_chess_data/lczero/medium"
    elif get_local_machine() == "gracjan":
        DEFAULT_LC_ZERO_WEIGHTS_PATH = "/home/gracjan/PycharmProjects/leela_weights/medium"
    elif get_local_machine() == "malgorzata":
        DEFAULT_LC_ZERO_WEIGHTS_PATH = "/home/gosia/dane/subgoal_chess_data/lczero/medium/medium"
else:
    DEFAULT_LC_ZERO_WEIGHTS_PATH = "/lcezero_weights/medium"
