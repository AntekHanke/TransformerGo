"""Stockfish-related functions."""
import chess
import chess.engine

from data_structures.data_structures import ImmutableBoard


def stockfish_evaluate_immutable_board(immutable_board: ImmutableBoard, time_limit: float = 0.05) -> float:
    engine = chess.engine.SimpleEngine.popen_uci("stockfish")
    result = engine.analyse(immutable_board.to_board(), chess.engine.Limit(time=time_limit))
    return result['score']

