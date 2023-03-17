import copy
import os
import random
from abc import ABC, abstractmethod
from datetime import date, datetime
from typing import Optional, List, Tuple, Set

import chess
import chess.engine
import numpy as np

from chess_engines.third_party.stockfish import StockfishEngine
from data_structures.data_structures import ImmutableBoard, HistoryLength
from policy.chess_policy import BasicChessPolicy, LCZeroPolicy
from policy.cllp import CLLP
from subgoal_generator.subgoal_generator import BasicChessSubgoalGenerator
from utils.chess960_conversion import chess960_to_standard
from utils.data_utils import immutable_boards_to_img


def log_engine_specific_info(s: str, path_to_log_dir: str, prefix: str = ">>> ") -> None:
    f = open(path_to_log_dir + "/log_engine_specific_info.txt", "a")
    f.write(prefix + s + "\n")
    f.close()


class ChessEngine(ABC):
    """Core class for chess engine"""

    @abstractmethod
    def new_game(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def propose_best_moves(
        self, current_state: chess.Board, number_of_moves: int, history: List[chess.Move]
    ) -> Optional[str]:
        raise NotImplementedError


class RandomChessEngine(ChessEngine):
    def __init__(self):
        self.name = "RANDOM"

    def propose_best_moves(
        self, current_state: chess.Board, number_of_moves: int = 0, history: List[chess.Move] = None
    ) -> Optional[str]:
        board = copy.deepcopy(current_state)
        moves: List[chess.Move] = list(board.legal_moves)
        best_move: str = np.random.choice(moves).uci()

        return best_move

    def new_game(self) -> None:
        pass


class PolicyChess(ChessEngine):
    def __init__(
        self,
        policy_checkpoint: Optional[str] = None,
        log_dir: str = None,
        debug_mode: bool = False,
        replace_legall_move_with_random: bool = False,
        do_sample: bool = True,
        name: str = "POLICY",
        use_lczero_policy: bool = False,
        history_length: HistoryLength = HistoryLength.no_history,
    ) -> None:

        self.name: str = name
        self.log_dir = log_dir
        self.debug_mode = debug_mode
        self.replace_legall_move_with_random = replace_legall_move_with_random
        self.do_sample = do_sample
        self.policy_checkpoint = policy_checkpoint
        self.use_lczero_policy = use_lczero_policy
        self.history_length = history_length
        self.chess_policy = None

    def new_game(self) -> None:

        today: datetime.date = date.today()
        now: datetime.date = datetime.now()
        self.log_dir = os.path.join(self.log_dir, str(today), f"{now.hour}_{now.minute}_{now.second}", f"{self.name}")
        os.makedirs(self.log_dir)

        if self.debug_mode:
            log_engine_specific_info(f"NAME OF ENGINE: {self.name}", self.log_dir)
            log_engine_specific_info(
                f"REPLACE LEGALL MOVE WITH RANDOM MOVE: {self.replace_legall_move_with_random}", self.log_dir
            )
            log_engine_specific_info(f"PATH TO CHECKPOINT OF POLICY: {self.policy_checkpoint}", self.log_dir)
            log_engine_specific_info(f"PATH TO FOLDER WITH ALL DATA OF ENGINE: {self.log_dir}", self.log_dir)

    def propose_best_moves(
        self, current_state: chess.Board, number_of_moves: int, history: List[chess.Move] = None
    ) -> Optional[str]:

        if self.chess_policy is None:
            if not self.use_lczero_policy:
                self.chess_policy = BasicChessPolicy(self.policy_checkpoint, self.history_length)
            else:
                self.chess_policy = LCZeroPolicy()

        if self.debug_mode:
            log_engine_specific_info("\n", self.log_dir)
            log_engine_specific_info(f"CURRENT STATE:{current_state.fen()}", self.log_dir)
            log_engine_specific_info(f"NUMBER OF GENERATED MOVES: {number_of_moves}", self.log_dir)
            log_engine_specific_info("JUST BEFORE SELECTING BEST MOVES", self.log_dir)

        try:
            moves, probs = self.chess_policy.get_best_moves(
                immutable_board=ImmutableBoard.from_board(current_state),
                history=None,
                num_return_sequences=number_of_moves,
                return_probs=True,
                do_sample=self.do_sample,
            )
            log_engine_specific_info(f"MOVES PROBABILITIES: {[int(10000*prob)/10000 for prob in probs]}", self.log_dir)
        except Exception as e:
            log_engine_specific_info(f"ERROR: {e}", self.log_dir)
            return None

        if self.debug_mode:
            log_engine_specific_info(f"AFTER SELECTING BEST MOVES. BEST MOVES: {moves}", self.log_dir)

        legall_moves: List[chess.Move] = list(current_state.legal_moves)
        print(f"FEN = {current_state.fen()}")
        converted_moves = [chess960_to_standard(move, current_state) for move in moves]
        print(f"POLICY {moves}: conv_moves: {converted_moves} | legall moves: {legall_moves}")


        for move in converted_moves:
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
    def __init__(
        self,
        name: str,
        generator_checkpoint: str,
        cllp_checkpoint: str,
        n_subgoals: int,
        stockfish_depth: int,
        log_dir: str,
        debug_mode: bool = False,
        replace_legall_move_with_random: bool = False,
    ) -> None:
        self.name = name
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
                f"REPLACE LEGALL MOVE WITH RANDOM MOVE: {self.replace_legall_move_with_random}", self.log_dir
            )
            log_engine_specific_info(f"PATH TO CHECKPOINT OF GENERATOR: {self.generator_checkpoint}", self.log_dir)
            log_engine_specific_info(f"PATH TO CHECKPOINT OF CLLP: {self.cllp_checkpoint}", self.log_dir)
            log_engine_specific_info(f"PATH TO FOLDER WITH ALL DATA OF ENGINE: {self.log_dir}", self.log_dir)
            log_engine_specific_info(f"STOCKFISH ENGINE DEPTH: {self.stockfish_depth}", self.log_dir)
            log_engine_specific_info(f"SET NUMBER OF GENERATED SUBGOALS: {self.n_subgoals}", self.log_dir)

    def choose_move_idx(
        self, move_values: List[float], moves: List[chess.Move], active_player: str, current_state: chess.Board
    ) -> Tuple[Optional[chess.Move], Optional[int]]:
        log_engine_specific_info("==========================================================", self.log_dir)
        log_engine_specific_info(str(move_values), self.log_dir)
        log_engine_specific_info(str(moves), self.log_dir)
        log_engine_specific_info("=========================================================", self.log_dir)
        legal_moves: Set[chess.Move] = set(current_state.legal_moves)
        legal_moves_960: Set[chess.Move] = set(chess.Board(current_state.fen(), chess960=True).legal_moves)
        sorted_moves = np.argsort(move_values)

        for i in range(len(move_values)):
            if active_player == "w":
                move = moves[sorted_moves[-(i + 1)]]
            else:
                move = moves[sorted_moves[i]]
            if move in legal_moves.union(legal_moves_960):
                log_engine_specific_info("-------------------------------------------------", self.log_dir)
                log_engine_specific_info(move.uci(), self.log_dir)
                log_engine_specific_info(str(legal_moves_960.difference(legal_moves)), self.log_dir)
                if move in legal_moves_960.difference(legal_moves):
                    move = legal_moves.difference(legal_moves_960).pop()
                    log_engine_specific_info(str(move), self.log_dir)
                return move, i

        return None, None

    def subgoal_filter(
        self, subgoals: List[ImmutableBoard], subgoals_valus: List[Optional[float]]
    ) -> Tuple[List[ImmutableBoard], List[float]]:
        subgoals_and_values: List[Tuple[ImmutableBoard, Optional[float]]] = list(zip(subgoals, subgoals_valus))
        subgoals_and_values = [sub_and_val for sub_and_val in subgoals_and_values if sub_and_val[1] is not None]
        filtred_subgoals: List[ImmutableBoard] = [sub[0] for sub in subgoals_and_values]
        filterd_values: List[float] = [val[1] for val in subgoals_and_values]

        if len(filtred_subgoals) < self.n_subgoals:
            new_subgoals = copy.deepcopy(filtred_subgoals)
            new_value = copy.deepcopy(filterd_values)
            while len(new_subgoals) < self.n_subgoals:
                new_subgoals.append(filtred_subgoals[0])
                new_value.append(filterd_values[0])
            return new_subgoals, new_value

        return filtred_subgoals, filterd_values

    def propose_best_moves(
        self, current_state: chess.Board, number_of_moves: int, history: List[chess.Move] = None
    ) -> Optional[str]:
        self.n_moves += 1
        immutable_current: ImmutableBoard = ImmutableBoard.from_board(current_state)
        batch_to_predict: List[Tuple[ImmutableBoard, ImmutableBoard]] = []

        subgoals: List[ImmutableBoard] = self.generator.generate_subgoals(
            ImmutableBoard.from_board(current_state), self.n_subgoals
        )
        subgoal_values: List[float] = self.stockfish.evaluate_boards_in_parallel(subgoals)

        subgoals, subgoal_values = self.subgoal_filter(subgoals, subgoal_values)

        for subgoal in subgoals:
            batch_to_predict.append((immutable_current, subgoal))

        if self.debug_mode:
            log_engine_specific_info("\n", self.log_dir)
            log_engine_specific_info(f"CURRENT STATE:{current_state.fen()}", self.log_dir)
            log_engine_specific_info(f"NUMBER OF GENERATED SUBGOALS: {len(subgoals)}", self.log_dir)
            log_engine_specific_info(f"SUBGOALS: {subgoals}", self.log_dir)
            log_engine_specific_info(f"STOCKSFISH VALUE OF SUBGOALDS: {subgoal_values}", self.log_dir)
            log_engine_specific_info("JUST BEFORE PRODUCING MOVES BY CLLP", self.log_dir)

        log_engine_specific_info(f"BATCH TO PREDICT: {batch_to_predict}", self.log_dir)

        paths: List[List[chess.Move]] = self.cllp.get_batch_path(batch_to_predict)
        moves: List[chess.Move] = [path[0] for path in paths]

        if self.debug_mode:
            log_engine_specific_info(f"AFTER PRODUCING MOVES BY CLLP. MOVES: {moves}", self.log_dir)

        if self.debug_mode:
            log_engine_specific_info("JUST BEFORE SELECTING BEST MOVE", self.log_dir)

        best_move, i = self.choose_move_idx(subgoal_values, moves, immutable_current.active_player, current_state)

        if self.debug_mode:
            log_engine_specific_info(f"AFTER SELECTING BEST MOVE: BEST MOVE {best_move}", self.log_dir)

        legal_moves: List[chess.Move] = list(current_state.legal_moves)

        if self.replace_legall_move_with_random:
            if best_move is None:
                best_move = random.choice(legal_moves)
                if self.debug_mode:
                    log_engine_specific_info(
                        f"THERE IS NO LEGALL MOVES (BY USING CLLP). USING A RANDOM MOVE FROM THE LITS OF LEAGL MOVES: {best_move.uci()}",
                        self.log_dir,
                    )

        descriptions: List[str] = [f"input move[{i}] = {best_move} l = {best_move in legal_moves}"]
        for i in range(self.n_subgoals):
            descriptions.append(f"val {subgoal_values[i]} moves = {[str(x) for x in paths[i]]}")

        fig = immutable_boards_to_img([ImmutableBoard.from_board(current_state)] + subgoals, descriptions)
        fig.savefig(os.path.join(self.log_dir, f"subgoal_{self.n_moves}.png"))

        return best_move.uci()
