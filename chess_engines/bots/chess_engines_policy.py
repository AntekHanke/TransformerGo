import copy
from abc import ABC, abstractmethod
from typing import Optional, List

import chess
import chess.engine
import numpy as np

from data_structures.data_structures import ImmutableBoard
from policy.chess_policy import BasicChessPolicy


def log(s: str, prefix: str = ">>> ") -> None:
    f = open("logs_inside_engine.txt", "a")
    f.write(prefix + s + "\n")
    f.close()


class ChessEngine(ABC):
    @abstractmethod
    def policy(self, current_state: chess.Board) -> str:
        pass


class RandomChessEngine(ChessEngine):
    def __init__(self) -> None:
        self.name = "Random Chess Engine 2"
        self.debug = False

    def policy(self, current_state: chess.Board) -> str:
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
        self.engine = engine
        self.depth_limit = chess.engine.Limit(depth=30)
        self.name = "Stockfish Engine"
        self.debug = False

    def policy(self, current_state: chess.Board) -> str:
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


    def __init__(self) -> None:
        self.name = "Policy Engine"
        POLICY_CHECKPOINT = "/home/tomasz/Research/subgoal_chess_data/local_leela_models/policy/final_model"
        self.chess_policy = BasicChessPolicy(POLICY_CHECKPOINT)

    def policy(self, current_state: chess.Board) -> str:
        move =  self.chess_policy.get_best_move(ImmutableBoard.from_board(current_state)).uci()
        log(f"Pawn promotion: current move {move}")
        return move

class CLLPChess(ChessEngine):
    def __init__(self, engine) -> None:
        self.engine = engine

    def policy(self, current_state: chess.Board) -> str:
        pass
