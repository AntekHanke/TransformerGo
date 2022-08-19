"""Stockfish-related functions."""
import chess
import chess.engine

from configs.global_config import VALUE_FOR_MATE
from data_structures.data_structures import ImmutableBoard


def stockfish_evaluate_immutable_board(immutable_board: ImmutableBoard, time_limit: float = 0.05) -> float:
    engine = chess.engine.SimpleEngine.popen_uci("stockfish")
    result = engine.analyse(immutable_board.to_board(), chess.engine.Limit(time=time_limit))['score']
    if not result.is_mate():
        return result.relative.cp
    else:
        #TODO(TO): check if signs are correct
        if immutable_board.active_player == "w":
            return -VALUE_FOR_MATE
        elif immutable_board.active_player == "b":
            return VALUE_FOR_MATE