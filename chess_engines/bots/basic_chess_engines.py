import copy
import os
import random
from abc import ABC, abstractmethod
from datetime import date, datetime
from typing import Optional, List

import chess
import chess.engine
import numpy as np

from chess_engines.third_party.stockfish import StockfishEngine
from data_processing.data_utils import immutable_boards_to_img
from data_structures.data_structures import ImmutableBoard
from policy.chess_policy import BasicChessPolicy
from policy.cllp import CLLP
from subgoal_generator.subgoal_generator import BasicChessSubgoalGenerator


def log(s: str, prefix: str = ">>> ") -> None:
    f = open("logs_inside_engine.txt", "a")
    f.write(prefix + s + "\n")
    f.close()


class ChessEngine(ABC):
    def __init__(self, name: str, log_dir: str):
        self.name = name
        self.log_dir = log_dir

        self.current_game = {"move": [], "board": []}

    def new_game(self):
        today = date.today()
        now = datetime.now()
        self.log_dir = os.path.join(self.log_dir, str(today), f"{now.hour}_{now.minute}_{now.second}")

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
    def __init__(self, log_dir: str) -> None:
        super().__init__("Random Chess Engine", log_dir)
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
    def __init__(self, engine, log_dir: str) -> None:
        super().__init__("Stockfish Chess Engine", log_dir)
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
    def __init__(self, policy_checkpoint, log_dir: str) -> None:
        super().__init__("Policy Engine", log_dir)
        self.chess_policy = BasicChessPolicy(policy_checkpoint)

    def propose_best_move(self, current_state: chess.Board) -> str:
        move = self.chess_policy.get_best_move(ImmutableBoard.from_board(current_state)).uci()
        return move


class SubgoalWithCLLP(ChessEngine):
    def __init__(self, generator_checkpoint: str, cllp_checkpoint: str, log_dir: str) -> None:
        super().__init__("One subgoal with CLLP", log_dir)
        self.generator = BasicChessSubgoalGenerator(generator_checkpoint)
        self.cllp = CLLP(cllp_checkpoint)

    def propose_best_move(self, current_state: chess.Board) -> str:
        subgoal = self.generator.generate_subgoals(ImmutableBoard.from_board(current_state), 1)[0]
        move = self.cllp.get_path(ImmutableBoard.from_board(current_state), subgoal)[0].uci()
        return move


class SubgoalWithCLLPStockfish(ChessEngine):
    # TODO: code in progress
    def __init__(
        self, generator_checkpoint: str, cllp_checkpoint: str, n_subgoals: int, stockfish_depth: int, log_dir: str
    ) -> None:
        super().__init__(f"{n_subgoals} subgoals CLLP + Stockfish", log_dir)
        self.generator = BasicChessSubgoalGenerator(generator_checkpoint)
        self.cllp = CLLP(cllp_checkpoint)
        self.stockfish = StockfishEngine(depth_limit=stockfish_depth)

        self.n_subgoals = n_subgoals
        self.n_moves = 0

    def choose_move_idx(self, move_values, moves, active_player, legal_moves):
        sorted_moves = np.argsort(move_values)

        for i in range(len(move_values)):
            if active_player == "w":
                move = moves[sorted_moves[-(i + 1)]]
            else:
                move = moves[sorted_moves[i]]
            if move in legal_moves:
                return move, i

        else:
            raise Exception("No legal move found")

    def propose_best_move(self, current_state: chess.Board) -> str:
        self.n_moves += 1

        subgoals = self.generator.generate_subgoals(ImmutableBoard.from_board(current_state), self.n_subgoals)
        subgoal_values = self.stockfish.evaluate_boards_in_parallel(subgoals)

        batch_to_predict = []
        for subgoal in subgoals:
            batch_to_predict.append((ImmutableBoard.from_board(current_state), subgoal))
        paths = self.cllp.get_paths_batch(batch_to_predict)
        sorted_moves = np.argsort(subgoal_values)
        moves = [path[0].uci() for path in paths]

        immutable_current = ImmutableBoard.from_board(current_state)

        legal_moves = [x.uci() for x in current_state.legal_moves]
        move, i = self.choose_move_idx(subgoal_values, moves, immutable_current.active_player, legal_moves)
        # move = self.cllp.get_path(ImmutableBoard.from_board(current_state), best_subgoal)[0].uci()

        is_legal = move in legal_moves

        descriptions = [f"input move[{i}] = {move} l = {move in legal_moves}"]
        for i in range(self.n_subgoals):
            descriptions.append(f"val {subgoal_values[i]} moves = {[str(x) for x in paths[i]]}")

        fig = immutable_boards_to_img([ImmutableBoard.from_board(current_state)] + subgoals, descriptions)

        fig.savefig(
            os.path.join(self.log_dir, f"subgoal_{self.n_moves}.png")
        )

        return move
