"""Stockfish-related functions."""
import chess
import chess.engine

from configs.global_config import VALUE_FOR_MATE
from data_structures.data_structures import ImmutableBoard


class StockfishEngine:
    def __init__(self):
        self.engine = chess.engine.SimpleEngine.popen_uci("stockfish")

    def evaluate_immutable_board(self, immutable_board: ImmutableBoard, time_limit: float = 0.05) -> float:
        # engine = chess.engine.SimpleEngine.popen_uci("stockfish")
        result = self.engine.analyse(immutable_board.to_board(), chess.engine.Limit(time=time_limit))['score']
        # self.engine.quit()
        # self.engine.close()

        if not result.is_mate():
            return result.relative.cp
        else:
            #TODO(TO): check if signs are correct
            if immutable_board.active_player == "w":
                return -VALUE_FOR_MATE
            elif immutable_board.active_player == "b":
                return VALUE_FOR_MATE


# x = StockfishEngine()
# for _ in range(100):
#     print(x.evaluate_immutable_board(ImmutableBoard.from_board(chess.Board())))