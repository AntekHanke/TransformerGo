import copy
import random
from abc import ABC, abstractmethod
from typing import Optional, List

import chess
import chess.engine
import numpy as np

from data_structures.data_structures import ImmutableBoard
from policy.chess_policy import BasicChessPolicy
from policy.cllp import CLLP


def log(s: str, prefix: str = ">>> ") -> None:
    f = open("logs_inside_engine.txt", "a")
    f.write(prefix + s + "\n")
    f.close()


class ChessEngine(ABC):
    def __init__(self, name: str):
        self.name = name

    def policy(self, current_state: chess.Board) -> str:
        legal_moves = [x.uci() for x in current_state.legal_moves]
        proposed_move = self.propose_best_move(current_state)
        if proposed_move in legal_moves:
            return proposed_move
        else:
            return random.choice(legal_moves)

    def propose_best_move(self, current_state: chess.Board) -> str:
        raise NotImplementedError


class RandomChessEngine(ChessEngine):
    def __init__(self) -> None:
        super().__init__("Random Chess Engine")
        self.debug = False

    def propose_best_move(self, current_state: chess.Board) -> str:
        board = copy.deepcopy(current_state)
        moves: List[chess.Move] = list(board.legal_moves)
        best_move: str = np.random.choice(moves).uci()

        if self.debug:
            log("current board: " + board.fen())
            log("best move: " + best_move)

        if best_move[-1] in {"n", "r", "b"}:
            best_move = best_move[:-1] + "q"
            if self.debug:
                current_move: str = best_move
                log(f"Pawn promotion: current move {current_move} ---> {best_move}")

        return best_move


class StockfishChessEngine(ChessEngine):
    def __init__(self, engine) -> None:
        super().__init__("Stockfish Chess Engine")
        self.engine = engine
        self.depth_limit = chess.engine.Limit(depth=30)
        self.debug = False

    def propose_best_move(self, current_state: chess.Board) -> str:
        board = copy.deepcopy(current_state)
        available_moves: List[chess.Move] = list(board.legal_moves)
        best_move: chess.Move = self.engine.analyse(board, limit=self.depth_limit, multipv=1)[0]["pv"]

        if best_move not in available_moves:
            best_move = np.random.choice(available_moves)

        if self.debug:
            log("current board: " + board.fen())
            log("best move: " + best_move.uci())

        return best_move.uci()


class PolicyChess(ChessEngine):
    def __init__(self, policy_checkpoint) -> None:
        super().__init__("Policy Engine")
        self.chess_policy = BasicChessPolicy(policy_checkpoint)

    def propose_best_move(self, current_state: chess.Board) -> str:
        move = self.chess_policy.get_best_move(ImmutableBoard.from_board(current_state)).uci()
        return move


class CLLPChess(ChessEngine):
    def __init__(self, generator_checkpoint: str, cllp_checkpoint: str) -> None:
        super().__init__("One subgoal with CLLP")
        self.generator = BasicChessPolicy(generator_checkpoint)
        self.cllp = CLLP(cllp_checkpoint)

    def policy(self, current_state: chess.Board) -> str:
        pass
