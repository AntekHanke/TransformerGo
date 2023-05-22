from typing import Type

import chess

from chess_engines.bots.basic_chess_engines import ChessEngine
from chess_engines.third_party.stockfish import StockfishEngine
from data_structures.data_structures import ImmutableBoard
from jobs.core import Job
from metric_logging import log_object, log_value, log_value_to_average, accumulator_to_logger
from utils.data_utils import immutable_boards_to_img
from value.chess_value import LCZeroValue


class GameBetweenEngines(Job):
    def __init__(
        self,
        engine_white_class: Type[ChessEngine],
        engine_black_class: Type[ChessEngine],
        engine_white_params: dict,
        engine_black_params: dict,
        eval_stockfish_path: str = None,
        eval_stockfish_depth: int = 20,
        check_best_moves_with_stockfish: bool = True,
    ):
        self.engine_white = engine_white_class(**engine_white_params)
        self.engine_black = engine_black_class(**engine_black_params)
        self.eval_stockfish = StockfishEngine(stockfish_path=eval_stockfish_path, depth_limit=eval_stockfish_depth)
        self.check_best_moves_with_stockfish = check_best_moves_with_stockfish
        self.eval_lc0 = LCZeroValue()
        self.players = {"white": self.engine_white, "black": self.engine_black}
        log_object("Players", f"White: {self.players['white'].name}, Black: {self.players['black'].name}")

    def execute(self):
        self.engine_white.new_game()
        self.engine_black.new_game()
        board = chess.Board()
        moves_counter = 0
        while not board.is_game_over():
            color = "white" if board.turn == chess.WHITE else "black"
            board_before_move = ImmutableBoard.from_board(board)
            engine = self.players[color]
            move = engine.propose_best_moves(current_state=board, number_of_moves=1)
            move_description = f"Move {moves_counter} by {engine.name}: {move}"
            log_object("Move", move_description)
            board.push(chess.Move.from_uci(move))
            stockfish_eval = self.eval_stockfish.evaluate_immutable_board(ImmutableBoard.from_board(board))
            leela_eval = self.eval_lc0.evaluate_immutable_board(ImmutableBoard.from_board(board))
            log_value("Stockfish evaluation", moves_counter, stockfish_eval)
            log_value("Leela evaluation", moves_counter, leela_eval)
            fig = immutable_boards_to_img([board_before_move], [f"{moves_counter} ({color}) m: {move} m_v: {stockfish_eval}"])
            log_object("Game", fig)
            moves_counter += 1
        log_object("Result", board.result())
