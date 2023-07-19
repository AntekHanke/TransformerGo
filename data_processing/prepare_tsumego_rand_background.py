import os
import random
from typing import Generator, Callable

import numpy as np
import pandas as pd
import sente

from sente import sgf, Game
from sente.exceptions import IllegalMoveException

from data_processing.prepare_tsumego import (
    numpy_to_sente_game,
    half_and_half_eyes,
    one_and_half_eyes,
)
from data_processing.rate_tsumego import rate_tsumego


def sente_game_to_numpy(game: Game):
    moves = game.get_default_sequence()
    array_board = np.zeros((19, 19))

    for move in moves:
        if move.get_stone() == sente.BLACK:
            array_board[move.get_x() - 1, move.get_y() - 1] = 1
        elif move.get_stone() == sente.WHITE:
            array_board[move.get_x() - 1, move.get_y() - 1] = -1

    return array_board


def background_generator(game_paths: str, game_paths_file: str) -> Generator[pd.DataFrame, None, None]:
    with open(os.path.join(game_paths, game_paths_file), "r") as f:
        file_paths = f.readlines()
    random.shuffle(file_paths)
    for path in file_paths:
        game = sgf.load(os.path.join(game_paths, path[:-1]))
        move_sequence = game.get_default_sequence()[:10]
        game.play_sequence(move_sequence)
        array_board = sente_game_to_numpy(game)
        yield array_board
    return None


def make_tsumego(
    tsumego: Callable,
    background: pd.DataFrame,
    start_x: int,
    start_y: int,
    half_eye_direction: int,
    distance: int,
    who_inside: int,
    who_to_move: int,
) -> dict:
    (
        np_array,
        coordinates_good,
        coordinates_bad,
        tsumego_type,
        tsumego_subtype,
        tsumego_exact,
        locality,
    ) = tsumego(
        np_array=background,
        start=(start_y, start_x),
        distance=distance,
        who_inside=who_inside,
        half_eye_direction=half_eye_direction,
    )
    game = numpy_to_sente_game(np_array, who_to_move=who_to_move)
    role = "Attack" if who_to_move == 1 else "Defense"
    tsumego_dict = {
        "Game": game,
        "Coordinates_good": coordinates_good,
        "Coordinates_bad": coordinates_bad,
        "Type": tsumego_type,
        "Subtype": tsumego_subtype,
        "Role": role,
        "Exact": tsumego_exact,
        "Locality": locality,
    }
    return tsumego_dict


def generate_tsumego(game_paths: str, game_paths_file: str) -> pd.DataFrame:
    generator = background_generator(game_paths, game_paths_file)
    tsumego_list = []

    for start_x in range(3, 10):
        for start_y in range(5, 10):
            for half_eye_direction in range(4):
                for distance in range(2, 13 - start_x):
                    if distance == 3 and half_eye_direction == 1:
                        continue
                    for who_to_move in [-1, 1]:
                        while True:
                            try:
                                new_row = make_tsumego(
                                    one_and_half_eyes,
                                    next(generator),
                                    start_x,
                                    start_y,
                                    half_eye_direction,
                                    distance,
                                    -1,
                                    who_to_move,
                                )
                                tsumego_list.append(new_row)
                                break
                            except IllegalMoveException:
                                pass

    for start_x in range(3, 10):
        for start_y in range(5, 10):
            for half_eye_direction in range(4):
                for distance in range(3, 13 - start_x):
                    if distance <= 3 and half_eye_direction % 2 == 1:
                        continue
                    for who_to_move in [-1, 1]:
                        new_row = make_tsumego(
                            half_and_half_eyes,
                            next(generator),
                            start_x,
                            start_y,
                            half_eye_direction,
                            distance,
                            -1,
                            who_to_move,
                        )
                        tsumego_list.append(new_row)

    tsumego_df = pd.DataFrame(tsumego_list)
    tsumego_df = rate_tsumego(tsumego_df)
    tsumego_df.drop(tsumego_df["KataGo"] < 0.9, inplace=True)
    return tsumego_df