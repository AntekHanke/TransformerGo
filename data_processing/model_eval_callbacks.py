import random

import chess
from transformers import TrainerCallback

from metric_logging import log_value, log_value_without_step
from policy.policy import Policy



class PolicyEvalCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, model=None, **kwargs):
        policy = Policy(model)
        board = chess.Board()
        correct_moves = 0
        total_moves = 0

        while not board.is_game_over():
            legal_moves = board.generate_legal_moves()
            # print(f"Board to generate move: \n {board}")
            move = random.choice([move for move in legal_moves])
            policy_move = policy.get_best_move(board)
            if policy_move in legal_moves:
                correct_moves += 1
            total_moves += 1
            board.push(move)

        log_value_without_step("policy legal moves", value=correct_moves / total_moves)
