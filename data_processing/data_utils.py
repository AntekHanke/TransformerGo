from mpmath import mp, mpf, fmod
import hashlib

mp.dps = 50
TRAIN_TEST_SPLIT_SEED = 11


def hash_string_to_int(arg):
    arg = str(arg) + str(TRAIN_TEST_SPLIT_SEED)
    return int(hashlib.sha256(arg.encode("utf-8")).hexdigest(), 16) % 10**30


def hash_string_to_float(arg):
    assert mp.dps >= 50
    x = mpf(hash_string_to_int(arg))
    return fmod(x * mp.pi, 1)


def get_split(arg, train_eval_split):
    float_hash = hash_string_to_float(arg)
    if float_hash < train_eval_split:
        return "train"
    else:
        return "eval"


import chess.svg
import matplotlib.pyplot as plt

x = chess.svg.board(board=chess.Board())

import cairosvg
from PIL import Image
from io import BytesIO


def boards_to_img(boards, boards_descriptions, size=5):

    fig = plt.figure(figsize=(size * len(boards), size))

    for board, title, num in zip(boards, boards_descriptions, range(len(boards))):
        plt.clf()
        fig.add_subplot(1, len(boards), num + 1)
        board = chess.svg.board(board=board)
        img_png = cairosvg.svg2png(board)
        img = Image.open(BytesIO(img_png))
        ax = plt.gca()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_title(title)
        plt.imshow(img)
        return fig
    # plt.show()


# boards_to_img([x,x], ["a", "b"])
