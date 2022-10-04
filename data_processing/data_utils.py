import hashlib
from typing import Union, List

import chess.svg
import matplotlib.pyplot as plt
import cairosvg

from mpmath import mp, mpf, fmod
from PIL import Image
from io import BytesIO

from configures.global_config import TRAIN_TEST_SPLIT_SEED
from data_structures.data_structures import ImmutableBoard
from metric_logging import log_object

mp.dps = 50

RESULT_TO_WINNER = {"1-0": "w", "0-1": "b", "1/2-1/2": None}


def hash_string_to_int(arg: Union[str, int]) -> int:
    arg = str(arg) + str(TRAIN_TEST_SPLIT_SEED)
    return int(hashlib.sha256(arg.encode("utf-8")).hexdigest(), 16) % 10**30


def hash_string_to_float(arg: Union[str, int]) -> float:
    assert mp.dps >= 50
    x = mpf(hash_string_to_int(arg))
    return fmod(x * mp.pi, 1)


def get_split(arg: Union[str, int], train_eval_split: float) -> str:
    float_hash = hash_string_to_float(arg)
    if float_hash < train_eval_split:
        return "train"
    else:
        return "eval"


def immutable_boards_to_img(
    immutable_boards: List[ImmutableBoard], descriptions: List[str], size: int = 5
) -> plt.Figure:

    fig = plt.figure(figsize=(size * len(immutable_boards), size))
    for immutable_board, title, num in zip(immutable_boards, descriptions, range(len(immutable_boards))):
        fig.add_subplot(1, len(immutable_boards), num + 1)
        immutable_board = chess.svg.board(board=immutable_board.to_board())
        img_png = cairosvg.svg2png(immutable_board)
        img = Image.open(BytesIO(img_png))
        ax = plt.gca()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_title(title)
        plt.imshow(img)
    return fig
