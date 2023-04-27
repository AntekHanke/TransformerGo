import time

import numpy as np
rng = np.random.default_rng(int(time.time()))

from utils.detect_local_machine import get_local_machine, is_local_machine

if is_local_machine():
    if get_local_machine() == "tomasz":
        NEPTUNE_API_TOKEN = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIwZDRhNWIxNS1hN2RkLTQ1ZjMtOGRmZi02MWI4NGRkZjA5MGMifQ=="

    elif get_local_machine() == "gracjan":
        NEPTUNE_API_TOKEN = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI2ZmZjYmYxOS1mNGFmLTQzNzktOWU5NC00NzQyNDYyZGEyZGMifQ=="

    elif get_local_machine() == "malgorzata":
        NEPTUNE_API_TOKEN = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIyZjIzMTAwNi0xZTkxLTQ1MWQtOTkzYy1mZGIzMGJmODk0NTYifQ=="

    elif get_local_machine() == "antek":
        NEPTUNE_API_TOKEN = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIyNzdhMDQzNi0yOTJmLTQ3MzAtYmU1Yi1hYTgzNWEwNzUyZjMifQ=="
    elif get_local_machine() == "mgrot":
        NEPTUNE_API_TOKEN = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI0YTRhYTg1YS0yZjdiLTRlNWItYWVhMC1lNjg2ZTQ2NTRkOGIifQ==" 
#NEPTUNE_PROJECT = "pmtest/subgoal-chess"
NEPTUNE_PROJECT = "mgrotkowski/debug-project" 

EAGLE_DATASET = ""
ENTROPY_HOME = ""
EAGLE_HOME = ""

TRAIN_TEST_SPLIT_SEED = 11
VALUE_FOR_MATE = 10000

MAX_JOBLIB_N_JOBS = 28
MAX_NEW_TOKENS_FOR_POLICY = 2
MAX_MOVES_FOR_CLLP = 6

MAX_GAME_LENGTH = 500

TOKENIZER = "board" # "board" or "pieces"

TOKENIZED_BOARD_LEN = 76 if TOKENIZER == "board" else 64

RANDOM_TOKENIZATION_ORDER = False # True or False

N_MOVES_HISTORY_FOR_MODEL_INPUT = 10

LCZERO_CLUSTER = "athena"  # "prometheus" or "athena"

NUMBER_OF_PRINT_SEPARATORS: int = 150

STOCKFISH_PATH = None if is_local_machine() else "/data_mg/stockfish/stockfish_15_linux_x64/stockfish_15_x64"

TREE_DISPLAY_SCORE_FACTOR = 1000
TREE_DISPLAY_LEVEL_DISTANCE_FACTOR = 1.2

LICHESS_ACCOUNTS_TO_TOKENS = {
    "subgoalsearchchess2": "lip_JwrHpEw7bVXS8WYpYeNJ",
    "subgoalchess_dev": "lip_kX19O4xorWT0WBtzmMmW",
    "martynawck": "lip_z5euZzVe9wzLgA5IWnnA"
}
