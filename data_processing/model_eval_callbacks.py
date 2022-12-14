import random

import chess
from transformers import TrainerCallback

from data_processing.chess_tokenizer import MoveDocodingException
from metric_logging import log_value, log_value_without_step
from policy.chess_policy import BasicChessPolicy



class PolicyEvalCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, model=None, **kwargs):
        policy = BasicPolicy(model)
        board = chess.Board()
        decoded_moves = 0
        legal_moves = 0
        total_moves = 0

        while not board.is_game_over():
            legal_moves = board.generate_legal_moves()
            # print(f"Board to generate move: \n {board}")
            move = random.choice([move for move in legal_moves])
            try:
                policy_move = policy.get_best_move(board)
                decoded_moves += 1
                if policy_move in legal_moves:
                    legal_moves += 1
                total_moves += 1
                board.push(move)
            except MoveDocodingException:
                total_moves += 1


        log_value_without_step("pgn_policy legal moves", value=legal_moves / total_moves)
        log_value_without_step("pgn_policy decoded moves", value=decoded_moves / total_moves)
        log_value_without_step("pgn_policy total moves", value=total_moves)
