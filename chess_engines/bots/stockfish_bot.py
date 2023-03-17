import math
from typing import Optional, List

import chess

from chess_engines.bots.basic_chess_engines import ChessEngine
from chess_engines.third_party.stockfish import StockfishEngine
from data_structures.data_structures import ImmutableBoard


class StockfishBotEngine(ChessEngine):
    def __init__(
        self,
        stockfish_depth: int = 5,
        stockfish_path: str = None,
    ):
        self.name = f"Stockfish depth {stockfish_depth}"
        self.stockfish_depth = stockfish_depth
        self.stockfish = StockfishEngine(stockfish_path=stockfish_path, depth_limit=stockfish_depth)


    def propose_best_moves(
        self, current_state: chess.Board, number_of_moves: int = 1, history: List[chess.Move] = None
    ) -> Optional[str]:
        immutable_state = ImmutableBoard.from_board(current_state)

        return [str(move) for move in self.stockfish.get_top_n_moves(immutable_state, number_of_moves)][0]

    def new_game(self) -> None:
        pass
