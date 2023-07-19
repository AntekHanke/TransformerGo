import numpy as np
import sente

eye = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
almost_eye = np.array([[1, 2, 1], [1, 0, 1], [1, 1, 1]])
five_square = np.array(
    [
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1],
    ]
)

ex_right_outside_black_free = np.array(
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 1, 1, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, -1, 0, -1, 0, -1, 0, -1, 1, -1, 1, 1, -1, -1, 0],
        [0, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0, -1, 0, -1, -1, -1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, -1, -1, -1, -1, -1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
)

### Directions: 0=up, 1=left, 2=down, 3=right

### 2 - play there
### 3 - dont play there


def one_and_half_eyes(
    np_array,
    distance=2,
    direction=(0, 1),
    half_eye_direction=0,
    start=(7, 7),
    who_inside=1,
):
    np_array[start[0] : start[0] + 5, start[1] : start[1] + 5] = (
        -five_square * who_inside
    )
    np_array[
        start[0] + direction[0] * distance : start[0] + 5 + direction[0] * distance,
        start[1] + direction[1] * distance : start[1] + 5 + direction[1] * distance,
    ] = (
        -five_square * who_inside
    )

    np_array[start[0] + 1 : start[0] + 4, start[1] + 1 : start[1] + 4] = (
        eye * who_inside
    )
    np_array[
        start[0] + direction[0] * distance + 1 : start[0] + 4 + direction[0] * distance,
        start[1] + direction[1] * distance + 1 : start[1] + 4 + direction[1] * distance,
    ] = (
        np.rot90(almost_eye, k=half_eye_direction) * who_inside
    )

    np_array[start[0] + 2, start[1] + 4 : start[1] + 4 + distance - 3] = who_inside
    np_array[start[0] + 1, start[1] + 4 : start[1] + 4 + distance - 3] = -who_inside
    np_array[start[0] + 3, start[1] + 4 : start[1] + 4 + distance - 3] = -who_inside

    indices_good = np.where(np.abs(np_array) == 2)
    coordinates_good = [
        (index[1] + 1, index[0] + 1) for index in zip(indices_good[0], indices_good[1])
    ]

    indices_bad = np.where(np.abs(np_array) == 3)
    coordinates_bad = [
        (index[1] + 1, index[0] + 1) for index in zip(indices_bad[0], indices_bad[1])
    ]

    type = "life_death"
    subtype = "simple_eyes"
    exact = "one_and_half_eyes"
    locality = distance + 5

    return np_array, coordinates_good, coordinates_bad, type, subtype, exact, locality


def half_and_half_eyes(
    np_array,
    distance=2,
    direction=(0, 1),
    half_eye_direction=0,
    start=(7, 7),
    who_inside=1,
):
    np_array[start[0] : start[0] + 5, start[1] : start[1] + 5] = (
        -five_square * who_inside
    )
    np_array[
        start[0] + direction[0] * distance : start[0] + 5 + direction[0] * distance,
        start[1] + direction[1] * distance : start[1] + 5 + direction[1] * distance,
    ] = (
        -five_square * who_inside
    )

    np_array[start[0] + 1 : start[0] + 4, start[1] + 1 : start[1] + 4] = (
        np.rot90(almost_eye, k=half_eye_direction) * who_inside
    )
    np_array[
        start[0] + direction[0] * distance + 1 : start[0] + 4 + direction[0] * distance,
        start[1] + direction[1] * distance + 1 : start[1] + 4 + direction[1] * distance,
    ] = (
        np.rot90(almost_eye, k=half_eye_direction) * who_inside
    )

    np_array[start[0] + 2, start[1] + 4 : start[1] + 4 + distance - 3] = who_inside
    np_array[start[0] + 1, start[1] + 4 : start[1] + 4 + distance - 3] = -who_inside
    np_array[start[0] + 3, start[1] + 4 : start[1] + 4 + distance - 3] = -who_inside

    indices_good = np.where(np.abs(np_array) == 3)
    coordinates_good = [
        (index[1] + 1, index[0] + 1) for index in zip(indices_good[0], indices_good[1])
    ]

    indices_bad = np.where(np.abs(np_array) == 2)
    coordinates_bad = [
        (index[1] + 1, index[0] + 1) for index in zip(indices_bad[0], indices_bad[1])
    ]

    type = "life_death"
    subtype = "simple_eyes"
    exact = "half_and_half_eyes"
    locality = distance + 5

    return np_array, coordinates_good, coordinates_bad, type, subtype, exact, locality


def numpy_to_sente_game(np_array, who_to_move=1):
    new_game = sente.Game()
    assert np_array.shape == (19, 19), "Wrong size of numpy array representing game"

    indices_black = np.where(np_array == 1)
    indices_white = np.where(np_array == -1)
    coordinates_black = list(zip(indices_black[0], indices_black[1]))
    coordinates_white = list(zip(indices_white[0], indices_white[1]))

    ##### Populate black stones ######
    for coor in coordinates_black:
        new_game.play(coor[1] + 1, coor[0] + 1)
        new_game.pss()

    new_game.pss()

    ##### Populate white stones ######
    for coor in coordinates_white:
        new_game.play(coor[1] + 1, coor[0] + 1)
        new_game.pss()

    if who_to_move == 1:
        new_game.pss()

    return new_game


from data_processing.goGraphics import plot_go_game

from data_processing.goPlay.goban import TransformerPolicy
from data_processing.goPlay.goban import ConvolutionPolicy
from data_structures.go_data_structures import GoImmutableBoard
import matplotlib.pyplot as plt
import pandas as pd
import os

save_plot_path = "./our_tsumego/"
if __name__ == '__main__':
    # game = numpy_to_sente_game(ex_right_outside_black_free)
    # print(game)

    tsumego = {
        "Game": [],
        "Coordinates_good": [],
        "Coordinates_bad": [],
        "Type": [],
        "Subtype": [],
        "Role": [],
        "Exact": [],
        "Locality": [],
    }

    tsumego_df = pd.DataFrame(tsumego)

    # t192k = TransformerPolicy("/mnt/c/Users/Antek/PycharmProjects/subgoal_search_chess/exclude/checkpoint-192500")
    # c365k = ConvolutionPolicy("/mnt/c/Users/Antek/PycharmProjects/subgoal_search_chess/exclude/checkpoint-365500")
    # ch351k = ConvolutionPolicy("/mnt/c/Users/Antek/PycharmProjects/subgoal_search_chess/exclude/checkpoint-351500", history=True, ignore_history=True)

    for start_x in range(3, 10):
        for start_y in range(5, 10):
            for half_eye_direction in range(4):
                for distance in range(2, 13 - start_x):
                    if distance == 3 and half_eye_direction == 1:
                        continue
                    for who_inside in [-1]:
                        # print(f"X: {start_x}, Y: {start_y}, half_eye_dir:{half_eye_direction}, dist:{distance}, who_ins:{who_inside}")
                        nex_right_outside_black_free = ex_right_outside_black_free.copy()
                        (
                            np_array,
                            coordinates_good,
                            coordinates_bad,
                            type,
                            subtype,
                            exact,
                            locality,
                        ) = one_and_half_eyes(
                            nex_right_outside_black_free,
                            start=(start_y, start_x),
                            distance=distance,
                            who_inside=who_inside,
                            half_eye_direction=half_eye_direction,
                        )
                        game = numpy_to_sente_game(np_array, who_to_move=1)

                        # sente.sgf.dump(game, os.path.join(save_plot_path,type+'_'+subtype+'_Attack_'+exact+'_'+str(half_eye_direction)+'_'+str(locality)+'.sgf'))

                        # a = ch351k.get_best_moves(GoImmutableBoard.from_game(game))
                        # b = t192k.get_best_moves(GoImmutableBoard.from_game(game))
                        # fig, ax = plot_go_game(game, explore_move_possibs=b)
                        # plt.savefig(os.path.join(save_plot_path, type+'_'+subtype+'_Attack_'+exact+'_'+str(half_eye_direction)+'_'+str(locality) + "_Transformer.png"))
                        # plt.close()
                        # plt.clf()

                        # fig, ax = plot_go_game(game, explore_move_possibs=a)
                        # plt.savefig(os.path.join(save_plot_path, type+'_'+subtype+'_Attack_'+exact+'_'+str(half_eye_direction)+'_'+str(locality) + "_Conv.png"))
                        # plt.close()
                        # plt.clf()

                        new_row = {
                            "Game": game,
                            "Coordinates_good": coordinates_good,
                            "Coordinates_bad": coordinates_bad,
                            "Type": type,
                            "Subtype": subtype,
                            "Role": "Attack",
                            "Exact": exact,
                            "Locality": locality,
                        }
                        tsumego_df = tsumego_df.append(new_row, ignore_index=True)
                        game = numpy_to_sente_game(np_array, who_to_move=-1)
                        new_row = {
                            "Game": game,
                            "Coordinates_good": coordinates_good,
                            "Coordinates_bad": coordinates_bad,
                            "Type": type,
                            "Subtype": subtype,
                            "Role": "Defend",
                            "Exact": exact,
                            "Locality": locality,
                        }
                        # print(game)
                        tsumego_df = tsumego_df.append(new_row, ignore_index=True)

    for start_x in range(3, 10):
        for start_y in range(5, 10):
            for half_eye_direction in range(4):
                for distance in range(3, 13 - start_x):
                    if distance <= 3 and half_eye_direction % 2 == 1:
                        continue
                    for who_inside in [-1]:
                        # print(f"X: {start_x}, Y: {start_y}, half_eye_dir:{half_eye_direction}, dist:{distance}, who_ins:{who_inside}")
                        nex_right_outside_black_free = ex_right_outside_black_free.copy()
                        (
                            np_array,
                            coordinates_good,
                            coordinates_bad,
                            type,
                            subtype,
                            exact,
                            locality,
                        ) = half_and_half_eyes(
                            nex_right_outside_black_free,
                            start=(start_y, start_x),
                            distance=distance,
                            who_inside=who_inside,
                            half_eye_direction=half_eye_direction,
                        )
                        game = numpy_to_sente_game(np_array, who_to_move=1)
                        new_row = {
                            "Game": game,
                            "Coordinates_good": coordinates_good,
                            "Coordinates_bad": coordinates_bad,
                            "Type": type,
                            "Subtype": subtype,
                            "Role": "Attack",
                            "Exact": exact,
                            "Locality": locality,
                        }
                        tsumego_df = tsumego_df.append(new_row, ignore_index=True)
                        game = numpy_to_sente_game(np_array, who_to_move=-1)
                        new_row = {
                            "Game": game,
                            "Coordinates_good": coordinates_good,
                            "Coordinates_bad": coordinates_bad,
                            "Type": type,
                            "Subtype": subtype,
                            "Role": "Defend",
                            "Exact": exact,
                            "Locality": locality,
                        }
                        # print(game)
                        tsumego_df = tsumego_df.append(new_row, ignore_index=True)

    # print(tsumego_df[['Coordinates_bad']])
