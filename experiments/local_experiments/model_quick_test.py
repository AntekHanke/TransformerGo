import random

import chess
import neptune.new as neptune

from configures.global_config import NEPTUNE_PROJECT, NEPTUNE_API_TOKEN, source_files_register
from data_processing.chess_tokenizer import ChessTokenizer

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
# weak_policy = Policy("/home/tomek/Research/subgoal_chess_data/fast_iter/policy_model")



move_number = 1

while not board.is_game_over():
    legal_moves = board.generate_legal_moves()
    print(f"Board to generate move: \n {board}")
    move = random.choice([move for move in legal_moves])

    encoded_move = ChessTokenizer.encode_move(move)
    decoded_move = ChessTokenizer.decode_move(encoded_move)
    print(f"Decoded move: {decoded_move} Original move: {[move.from_square, move.to_square]}")

    board.push(move)

    # policy_move = weak_policy.get_best_move(board)
    # print(f"Policy move: {policy_move}")

    # print(f"Move: {move}")
    print('************************')
    # if USE_NEPTUNE:
    #     image = boards_to_img([board], [f"Move: {move}"])
    #     run["gameplay"].log(image)
    #
    # # print(f"Board: {board} \n move: {move}")
    # board.push(move)

