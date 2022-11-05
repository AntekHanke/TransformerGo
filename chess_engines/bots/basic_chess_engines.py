import copy
import os
import random
from abc import ABC, abstractmethod
from datetime import date, datetime
from typing import Optional, List, Tuple

import chess
import chess.engine
import numpy as np

from chess_engines.third_party.stockfish import StockfishEngine
from data_processing.data_utils import immutable_boards_to_img
from data_structures.data_structures import ImmutableBoard
from policy.chess_policy import BasicChessPolicy
from policy.cllp import CLLP
from subgoal_generator.subgoal_generator import BasicChessSubgoalGenerator


def log_engine_specific_info(s: str, path_to_log_dir: str, prefix: str = ">>> ") -> None:
    f = open(path_to_log_dir + '/log_engine_specific_info.txt', "a")
    f.write(prefix + s + "\n")
    f.close()


class ChessEngine(ABC):
    """Core class for chess engine"""

    @abstractmethod
    def new_game(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def propose_best_moves(self, current_state: chess.Board, number_of_moves: int) -> Optional[str]:
        raise NotImplementedError


class PolicyChess(ChessEngine):
    def __init__(
            self, policy_checkpoint, log_dir: str, debug_mode: bool = False,
            replace_legall_move_with_random: bool = False
    ) -> None:

        self.name: str = "POLICY ENGINE"
        self.log_dir = log_dir
        self.debug_mode = debug_mode
        self.replace_legall_move_with_random = replace_legall_move_with_random
        self.policy_checkpoint = policy_checkpoint
        self.chess_policy = BasicChessPolicy(self.policy_checkpoint)

    def new_game(self) -> None:
        today: datetime.date = date.today()
        now: datetime.date = datetime.now()
        self.log_dir = os.path.join(self.log_dir, str(today), f"{now.hour}_{now.minute}_{now.second}", f"{self.name}")
        os.makedirs(self.log_dir)

        if self.debug_mode:
            log_engine_specific_info(f"NAME OF ENGINE: {self.name}", self.log_dir)
            log_engine_specific_info(
                f"REPLACE LEGALL MOVE WITH RANDOM STATE: {self.replace_legall_move_with_random}", self.log_dir
            )
            log_engine_specific_info(f"PATH TO CHECKPOINT OF POLICY: {self.policy_checkpoint}", self.log_dir)
            log_engine_specific_info(f"PATH TO FOLDER WITH ALL DATA OF ENGINE: {self.log_dir}", self.log_dir)

    def propose_best_moves(self, current_state: chess.Board, number_of_moves: int) -> Optional[str]:
        if self.debug_mode:
            log_engine_specific_info("\n", self.log_dir)
            log_engine_specific_info(f"CURRENT STATE:{current_state.fen()}", self.log_dir)
            log_engine_specific_info(f"NUMBER OF GENERATED MOVES: {number_of_moves}", self.log_dir)
            log_engine_specific_info("JUST BEFORE SELECTING BEST MOVES", self.log_dir)

        moves = self.chess_policy.get_best_moves(ImmutableBoard.from_board(current_state), number_of_moves)

        if self.debug_mode:
            log_engine_specific_info(f"AFTER SELECTING BEST MOVES. BEST MOVES: {moves}", self.log_dir)

        legall_moves: List[chess.Move] = list(current_state.legal_moves)
        for move in moves:
            if move in legall_moves:
                if self.debug_mode:
                    log_engine_specific_info(f"BEST MOVE CHOOSEN FROM LIST OF LEAGL MOVES: {move.uci()}", self.log_dir)
                return move.uci()
            else:
                if self.replace_legall_move_with_random:
                    move = random.choice(legall_moves)
                    if self.debug_mode:
                        log_engine_specific_info(
                            f"THERE IS NO LEGALL MOVES (BY USING POLICY). USING A RANDOM MOVE FROM THE LITS OF LEAGL MOVES: {move.uci()}"
                        )
                    return move.uci()

        if self.debug_mode:
            log_engine_specific_info(f"THERE IS NO LEGALL MOVES (BY USING POLICY): {None}", self.log_dir)
        return None


class SubgoalWithCLLPStockfish(ChessEngine):
    def __init__(self, generator_checkpoint: str, cllp_checkpoint: str, n_subgoals: int, stockfish_depth: int,
                 log_dir: str, debug_mode: bool = False,
                 replace_legall_move_with_random: bool = False) -> None:
        self.name: str = "POLICY SubgoalWithCLLPStockfish"
        self.generator_checkpoint = generator_checkpoint
        self.cllp_checkpoint = cllp_checkpoint
        self.n_subgoals = n_subgoals
        self.stockfish_depth = stockfish_depth
        self.log_dir = log_dir
        self.debug_mode = debug_mode
        self.replace_legall_move_with_random = replace_legall_move_with_random
        self.n_moves: int = 0

        self.generator = BasicChessSubgoalGenerator(generator_checkpoint)
        self.cllp = CLLP(cllp_checkpoint)
        self.stockfish = StockfishEngine(depth_limit=stockfish_depth)

    def new_game(self) -> None:
        today: datetime.date = date.today()
        now: datetime.date = datetime.now()
        self.log_dir = os.path.join(self.log_dir, str(today), f"{now.hour}_{now.minute}_{now.second}", f"{self.name}")
        os.makedirs(self.log_dir)

        if self.debug_mode:
            log_engine_specific_info(f"NAME OF ENGINE: {self.name}", self.log_dir)
            log_engine_specific_info(
                f"REPLACE LEGALL MOVE WITH RANDOM STATE: {self.replace_legall_move_with_random}", self.log_dir
            )
            log_engine_specific_info(f"PATH TO CHECKPOINT OF GENERATOR: {self.generator_checkpoint}", self.log_dir)
            log_engine_specific_info(f"PATH TO CHECKPOINT OF CLLP: {self.cllp_checkpoint}", self.log_dir)
            log_engine_specific_info(f"PATH TO FOLDER WITH ALL DATA OF ENGINE: {self.log_dir}", self.log_dir)
            log_engine_specific_info(f"STOCKFISH ENGINE DEPTH: {self.stockfish_depth}", self.log_dir)
            log_engine_specific_info(f"SET NUMBER OF GENERATED SUBGOALS: {self.n_subgoals}", self.log_dir)

    @staticmethod
    def choose_move_idx(move_values: List[float], moves: List[chess.Move], active_player: str, legal_moves: List[chess.Move]) -> Tuple[Optional[chess.Move], Optional[int]]:
        sorted_moves = np.argsort(move_values)

        for i in range(len(move_values)):
            if active_player == "w":
                move = moves[sorted_moves[-(i + 1)]]
            else:
                move = moves[sorted_moves[i]]
            if move in legal_moves:
                return move, i

        return None, None

    def propose_best_moves(self, current_state: chess.Board, number_of_moves: int) -> Optional[str]:
        self.n_moves += 1
        immutable_current: ImmutableBoard = ImmutableBoard.from_board(current_state)
        batch_to_predict: List[Tuple[ImmutableBoard, ImmutableBoard]] = []

        subgoals: List[ImmutableBoard] = self.generator.generate_subgoals(ImmutableBoard.from_board(current_state),
                                                                          self.n_subgoals)
        subgoal_values: List[float] = self.stockfish.evaluate_boards_in_parallel(subgoals)

        for subgoal in subgoals:
            batch_to_predict.append((immutable_current, subgoal))

        if self.debug_mode:
            log_engine_specific_info("\n", self.log_dir)
            log_engine_specific_info(f"CURRENT STATE:{current_state.fen()}", self.log_dir)
            log_engine_specific_info(f"NUMBER OF GENERATED SUBGOALS: {len(subgoals)}", self.log_dir)
            log_engine_specific_info(f"SUBGOALS: {subgoals}", self.log_dir)
            log_engine_specific_info(f"STOCKSFISH VALUE OF SUBGOALDS: {subgoal_values}", self.log_dir)
            log_engine_specific_info("JUST BEFORE PRODUCING MOVES BY CLLP", self.log_dir)

        paths: List[List[chess.Move]] = self.cllp.get_batch_path(batch_to_predict)
        moves: List[chess.Move] = [path[0] for path in paths]
        legal_moves: List[chess.Move] = list(current_state.legal_moves)

        if self.debug_mode:
            log_engine_specific_info(f"AFTER PRODUCING MOVES BY CLLP. MOVES: {moves}", self.log_dir)

        if self.debug_mode:
            log_engine_specific_info("JUST BEFORE SELECTING BEST MOVE", self.log_dir)

        best_move, i = self.choose_move_idx(subgoal_values, moves, immutable_current.active_player, legal_moves)

        if self.debug_mode:
            log_engine_specific_info(f"AFTER SELECTING BEST MOVE: BEST MOVE {best_move}", self.log_dir)

        if self.replace_legall_move_with_random:
            if best_move is None:
                best_move = random.choice(legal_moves)
            if self.debug_mode:
                log_engine_specific_info(
                    f"THERE IS NO LEGALL MOVES (BY USING CLLP). USING A RANDOM MOVE FROM THE LITS OF LEAGL MOVES: {best_move.uci()}", self.log_dir
                )

        descriptions: List[str] = [f"input move[{i}] = {best_move} l = {best_move in legal_moves}"]
        for i in range(self.n_subgoals):
            descriptions.append(f"val {subgoal_values[i]} moves = {[str(x) for x in paths[i]]}")

        fig = immutable_boards_to_img([ImmutableBoard.from_board(current_state)] + subgoals, descriptions)
        fig.savefig(os.path.join(self.log_dir, f"subgoal_{self.n_moves}.png"))

        return best_move.uci()
