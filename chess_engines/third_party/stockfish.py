"""Stockfish-related functions."""
from typing import List, Union, Tuple
from joblib import Parallel, delayed

import chess
import chess.engine

from configures.global_config import VALUE_FOR_MATE, MAX_JOBLIB_N_JOBS, STOCKFISH_PATH
from data_structures.data_structures import ImmutableBoard

DEFAULT_STOCKFISH_PATH = STOCKFISH_PATH
DEFAULT_STOCKFISH_PATH_CLUSTER = "/Stockfish/src/stockfish"


class StockfishEngine:
    """Wrapper for stockfish chess engine."""

    def __init__(self, stockfish_path: Union[str, None] = None, depth_limit: int = 15):

        if stockfish_path is None:
            print(f"Using default stockfish path.")
            stockfish_path = DEFAULT_STOCKFISH_PATH
        elif stockfish_path == "cluster":
            stockfish_path = DEFAULT_STOCKFISH_PATH_CLUSTER
        self.stockfish_path = stockfish_path

        self.depth_limit = depth_limit

    @staticmethod
    def get_result_score(result):
        return result.relative.score(mate_score=VALUE_FOR_MATE)

    @staticmethod
    def absolute_v(player, v):
        if player == "w":
            return v
        elif player == "b":
            return -v

    def evaluate_immutable_board(self, immutable_board: ImmutableBoard) -> Union[float, None]:
        """
        Function is used to evaluate the state of the game, after each evaluation the chess machine is reset.

        :param: immutable_board: Input chess board.
        :return: Evaluation of the state of the game.
        """
        engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
        player = immutable_board.active_player
        try:
            result = engine.analyse(
                immutable_board.to_board(), chess.engine.Limit(depth=self.depth_limit), game=object()
            )["score"]
            engine.quit()
            return StockfishEngine.absolute_v(player, self.get_result_score(result))
        except chess.engine.EngineTerminatedError:
            return None

    def get_top_n_moves(
        self, immutable_board: ImmutableBoard, top_n_moves: Union[int, None] = None
    ) -> Tuple[chess.Move]:

        board = immutable_board.to_board()
        move_scores = {move: None for move in board.legal_moves}

        boards_to_analyze = [immutable_board.act(move) for move in board.legal_moves]
        move_scores_list = self.evaluate_boards_in_parallel(boards_to_analyze)
        for move, score in zip(board.legal_moves, move_scores_list):
            move_scores[move] = score

        reverse_order = immutable_board.active_player == "w"
        sorted_moves, scores = zip(*sorted(move_scores.items(), key=lambda x: x[1], reverse=reverse_order))
        if top_n_moves is None:
            return sorted_moves
        else:
            return sorted_moves[:top_n_moves]

    def evaluate_boards_in_parallel(self, list_of_immutable_boards: List[ImmutableBoard]) -> List[float]:
        print(f"Parallel evaluation of {len(list_of_immutable_boards)} boards.")
        if len(list_of_immutable_boards) == 0:
            return []
        return Parallel(n_jobs=min(len(list_of_immutable_boards), MAX_JOBLIB_N_JOBS))(
            delayed(self.evaluate_immutable_board)(board) for board in list_of_immutable_boards
        )
