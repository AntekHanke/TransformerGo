"""Stockfish-related functions."""
from typing import List, Union
from joblib import Parallel, delayed

import chess
import chess.engine

from configs.global_config import VALUE_FOR_MATE, MAX_JOBLIB_N_JOBS
from data_structures.data_structures import ImmutableBoard

DEFAULT_STOCKFISH_PATH = "stockfish"
DEFAULT_STOCKFISH_PATH_CLUSTER = "/Stockfish/src/stockfish"


class StockfishEngine:
    """Wrapper for stockfish chess engine."""

    def __init__(self, stockfish_path: Union[str, None], depth_limit: int = 10):

        if stockfish_path is None:
            print(f"Using default stockfish path.")
            stockfish_path = DEFAULT_STOCKFISH_PATH
        elif stockfish_path == "cluster":
            stockfish_path = DEFAULT_STOCKFISH_PATH_CLUSTER
        self.stockfish_path = stockfish_path

        self.depth_limit = depth_limit

    def get_result_score(self, immutable_board, result):
        if not result.is_mate():
            return result.relative.cp
        else:
            if immutable_board.active_player == "w":
                return -VALUE_FOR_MATE
            elif immutable_board.active_player == "b":
                return VALUE_FOR_MATE

    def evaluate_immutable_board(self, immutable_board: ImmutableBoard) -> Union[float, None]:
        """
        Function is used to evaluate the state of the game, after each evaluation the chess machine is reset.

        :param: immutable_board: Input chess board.
        :param: time_limit: (probably) Time to assess the state of play by stockfish.
        :return: Evaluation of the state of the game.
        """
        engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
        try:
            result = engine.analyse(
                immutable_board.to_board(), chess.engine.Limit(depth=self.depth_limit), game=object()
            )["score"]
            engine.quit()
            return self.get_result_score(immutable_board, result)
        except chess.engine.EngineTerminatedError:
            return None

    def get_top_n_moves(
        self, immutable_board: ImmutableBoard, top_n_moves: Union[int, None] = None
    ) -> List[chess.Move]:

        # engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
        board = immutable_board.to_board()
        move_scores = {move: None for move in board.legal_moves}

        boards_to_analyze = [immutable_board.act(move) for move in board.legal_moves]
        move_scores_list = self.evaluate_boards_in_parallel(boards_to_analyze)
        for move, score in zip(board.legal_moves, move_scores_list):
            move_scores[move] = score

        sorted_moves, scores = zip(*sorted(move_scores.items(), key=lambda x: x[1], reverse=True))
        if top_n_moves is None:
            return sorted_moves
        else:
            return sorted_moves[:top_n_moves]

    def evaluate_boards_in_parallel(self, list_of_immutable_boards):
        print(f"Parallel evaluation of {len(list_of_immutable_boards)} boards.")
        if len(list_of_immutable_boards) == 0:
            return []
        return Parallel(n_jobs=min(len(list_of_immutable_boards), MAX_JOBLIB_N_JOBS))(
            delayed(self.evaluate_immutable_board)(board) for board in list_of_immutable_boards
        )
