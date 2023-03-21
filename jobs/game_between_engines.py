from pathlib import Path
from typing import Type, Tuple, List

import chess

from chess_engines.bots.basic_chess_engines import PolicyChess, ChessEngine
from chess_engines.bots.mcts_bot import MCTSChessEngine, VanillaMCTSChessEngine
from chess_engines.bots.stockfish_bot import StockfishBotEngine
from chess_engines.third_party.stockfish import StockfishEngine
from data_structures.data_structures import ImmutableBoard
from jobs.core import Job
from metric_logging import log_object, log_value
from utils.data_utils import immutable_boards_to_img


class GameBetweenEngines(Job):
    def __init__(
        self,
        engine_white_params: List[Type[ChessEngine], dict],
        engine_black_params: List[Type[ChessEngine], dict],
        eval_stockfish_path: str = None,
        eval_stockfish_depth: int = 20,
    ):
        self.engine_white = engine_white_params[0](**engine_white_params[1])
        self.engine_black = engine_black_params[0](**engine_black_params[1])

        self.eval_stockfish = StockfishEngine(stockfish_path=eval_stockfish_path, depth_limit=eval_stockfish_depth)

        self.players = {"w": self.engine_white, "b": self.engine_black}
        log_object("Players", f"White: {self.players['w'].name}, Black: {self.players['b'].name}")

    def execute(self):
        board = chess.Board()
        moves_counter = 0
        while not board.is_game_over():
            color = "w" if board.turn == chess.WHITE else "b"
            engine = self.players[color]
            player = engine.name

            move = engine.propose_best_moves(current_state=board, number_of_moves=1)
            move_description = f"Move {moves_counter} by {player}: {move}"
            print(move_description)
            log_object("Move", move_description)
            board.push(chess.Move.from_uci(move))
            stockfish_eval = self.eval_stockfish.evaluate_immutable_board(ImmutableBoard.from_board(board))
            log_value("Stockfish evaluation", moves_counter, stockfish_eval)
            fig = immutable_boards_to_img([ImmutableBoard.from_board(board)], [f"Move {moves_counter}"])
            log_object("Game", fig)

            moves_counter += 1

        log_object("Result", board.result())
