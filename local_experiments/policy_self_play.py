import chess
import neptune.new as neptune

from config.global_config import NEPTUNE_PROJECT, NEPTUNE_API_TOKEN, source_files_register
from data_processing.chess_tokenizer import ChessTokenizer
from data_processing.data_utils import boards_to_img
from policy.policy import Policy

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
weak_policy = Policy("/home/tomek/Research/subgoal_chess_data/local_policy")

move_number = 1

while not board.is_game_over():
    move = weak_policy.get_best_move(board)
    if USE_NEPTUNE:
        image = boards_to_img([board], [f"Move: {move}"])
        run["gameplay"].log(image)

    print(f"Board: {board} \n move: {move}")
    board.push(move)

