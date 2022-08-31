"""Stockfish-related functions."""
from typing import List, Dict, Union

import chess
import chess.engine

from configs.global_config import VALUE_FOR_MATE
from data_structures.data_structures import ImmutableBoard


class StockfishEngine:
    """Wrapper for stockfish chess engine."""

    @classmethod
    def get_result_score(cls, immutable_board, result):
        if not result.is_mate():
            return result.relative.cp
        else:
            if immutable_board.active_player == "w":
                return -VALUE_FOR_MATE
            elif immutable_board.active_player == "b":
                return VALUE_FOR_MATE

    def evaluate_immutable_board_by_stockfish_with_resret_machine(
        self, immutable_board: ImmutableBoard, depth_limit: int = 10
    ) -> float:
        """
        Function is used to evaluate the state of the game, after each evaluation the chess machine is reset.

        :param: immutable_board: Input chess board.
        :param: time_limit: (probabliy) Time to assess the state of play by stockfish.
        :return: Evaluation of the state of the game.
        """
        engine = chess.engine.SimpleEngine.popen_uci("stockfish")
        result = engine.analyse(immutable_board.to_board(), chess.engine.Limit(depth=depth_limit), game=object())[
            "score"
        ]
        engine.quit()

        return StockfishEngine.get_result_score(immutable_board, result)

    def get_top_n_moves(
        self, immutable_board: ImmutableBoard, n: Union[int, None] = None, depth_limit: int = 12
    ) -> List[chess.Move]:

        engine = chess.engine.SimpleEngine.popen_uci("stockfish")
        board = immutable_board.to_board()
        move_scores = {move: None for move in board.legal_moves}

        for move in move_scores:
            result = engine.analyse(board, chess.engine.Limit(depth=12), root_moves=[move])["score"]
            move_scores[move] = StockfishEngine.get_result_score(immutable_board, result)

        engine.quit()
        sorted_moves = sorted(move_scores.items(), key=lambda x: x[1], reverse=True)
        if n is None:
            return sorted_moves
        else:
            return sorted_moves[:n]
