import chess
import neptune.new as neptune

from configs.global_config import NEPTUNE_PROJECT, NEPTUNE_API_TOKEN, source_files_register
from data_structures.data_structures import ImmutableBoard
from data_processing.data_utils import immutable_boards_to_img
from policy.chess_policy import ChessPolicy

source_files_register.register(__file__)

USE_NEPTUNE = 0

if USE_NEPTUNE:
    run = neptune.init(
        name="weak policy self play",
        project=NEPTUNE_PROJECT,
        api_token=NEPTUNE_API_TOKEN,
        source_files=source_files_register.get()
    )


board = chess.Board()
chess_policy = ChessPolicy("/home/tomek/Research/subgoal_chess_data/fast_iter/policy_model")

move_number = 1

while not board.is_game_over():
    print(f"Board to generate move: \n {board}")
    move = chess_policy.get_best_move(ImmutableBoard.from_board(board))
    print(f"Move: {move}")
    print('************************')

    if USE_NEPTUNE:
        image = immutable_boards_to_img([ImmutableBoard.from_board(board)], [f"Move {move_number}: {move}"])
        run["gameplay"].log(image)

    print(f"Board: {board} \n move: {move}")
    board.push(move)

