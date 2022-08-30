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
            # TODO(TO): check if signs are correct
            if immutable_board.active_player == "w":
                return -VALUE_FOR_MATE
            elif immutable_board.active_player == "b":
                return VALUE_FOR_MATE


def evaluate_immutable_board_by_stockfish_with_resret_machine(immutable_board: ImmutableBoard,
                                                              depth_limit: int = 10
                                                              ) -> float:
    """
    Function is used to evaluate the state of the game, after each evaluation the chess machine is reset.

    :param: immutable_board: Input chess board.
    :param: time_limit: (probabliy) Time to assess the state of play by stockfish.
    :return: Evaluation of the state of the game.
    """
    engine = chess.engine.SimpleEngine.popen_uci("stockfish")
    result = engine.analyse(immutable_board.to_board(), chess.engine.Limit(depth=depth_limit), game=object())['score']
    engine.quit()

    if not result.is_mate():
        return result.relative.cp
    else:
        # TODO(TO): check if signs are correct
        if immutable_board.active_player == "w":
            return -VALUE_FOR_MATE
        elif immutable_board.active_player == "b":
            return VALUE_FOR_MATE
