import numpy as np
from random import sample

# Pieces are always designed for the left side

background_element_list = [
    np.array(
        [
            [0, 1, -1, 0, 0],
            [0, 1, -1, 1, 1],
            [0, 0, 1, -1, 0],
            [0, 1, 1, -1, 0],
            [0, 1, -1, -1, 0],
            [0, -1, 1, 0, 0],
            [0, -1, -1, -1, 0],
        ]
    ),
    np.array([[-1, -1], [0, -1], [-1, -1], [0, -1], [-1, -1]]),
    np.array(
        [
            [0, 0, -1, 0],
            [0, 0, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 0],
            [0, 0, 0, -1],
            [0, 0, 0, -1],
        ]
    ),
    np.array(
        [
            [0, 0, -1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, -1, 0, 1, 0],
            [0, 0, -1, 1, 0, 1],
            [0, 0, -1, 1, 0, 1],
            [0, 0, -1, 1, 0, 1],
            [0, 0, -1, 1, 0, 1],
            [0, 0, 0, 0, 1, 0],
        ]
    ),
    np.array(
        [
            [0, 0, 1, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
        ]
    ),
    np.array(
        [
            [0, 0, -1, 0],
            [0, 0, 0, 0],
            [0, 0, 0, -1],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, -1, 0],
        ]
    ),
]


def get_random_background() -> np.ndarray:
    board = np.zeros((19, 19))
    start_coordinates = {}
    for start_position in range(18):
        for side in range(4):
            start_coordinates[(start_position, side)] = [19 - start_position, 19]

    for _ in range(100):
        element = sample(background_element_list, 1)[0]
        element *= np.random.choice([-1, 1])
        if np.random.randint(0, 2):
            element = np.flip(element, 0)
        start = sample(start_coordinates.keys(), 1)[0]
        available_size = start_coordinates[start]
        if (
            available_size[0] >= element.shape[0]
            and available_size[1] >= element.shape[1]
        ):
            board = np.rot90(board, start[1])
            board[
                start[0] : start[0] + element.shape[0], 0 : element.shape[1]
            ] = element
            board = np.rot90(board, -start[1])
            for i in range(element.shape[0]):
                if (start[0] + i, start[1]) in start_coordinates.keys():
                    del start_coordinates[(start[0] + i, start[1])]
            for i in range(start[0]):
                if (i, start[1]) in start_coordinates.keys():
                    start_coordinates[(i, start[1])][0] = min(
                        start_coordinates[(i, start[1])][0], start[0]
                    )
            for i in range(element.shape[1]):
                if (18 - i, (start[1] - 1) % 4) in start_coordinates.keys():
                    start_coordinates[(18 - i, (start[1] - 1) % 4)][1] = min(start_coordinates[(18 - i, (start[1] - 1) % 4)][1], start[0])
                if (i, (start[1] + 1) % 4) in start_coordinates.keys():
                    start_coordinates[(i, (start[1] + 1) % 4)][1] = min(
                        start_coordinates[(i, (start[1] + 1) % 4)][1],
                        19 - start[0] - element.shape[0]
                    )
            for key in list(start_coordinates.keys()):
                coordinates = start_coordinates[key]
                if coordinates[0] < 2 or coordinates[1] < 5:
                    del start_coordinates[key]
            if not start_coordinates:
                break

    return board
