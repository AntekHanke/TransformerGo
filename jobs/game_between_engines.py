from pathlib import Path

import chess

from chess_engines.bots.basic_chess_engines import PolicyChess
from chess_engines.bots.mcts_bot import MCTSChessEngine
from chess_engines.bots.stockfish_bot import StockfishBotEngine
from chess_engines.third_party.stockfish import StockfishEngine
from data_structures.data_structures import ImmutableBoard
from jobs.core import Job
from metric_logging import log_object, log_value
from utils.data_utils import immutable_boards_to_img


class GameBetweenEngines(Job):
    def __init__(
        self,
        generator_path: str = None,
        cllp_path: str = None,
        out_dir: str = None,
        stockfish_path: str = None,
        stockfish_depth: int = 20,
        stockfish_bot_depth: int = 5,
        sort_subgoals_by: str = None,
    ):
        self.stockfish = StockfishEngine(stockfish_path=stockfish_path, depth_limit=stockfish_depth)
        self.engine_MCTS = MCTSChessEngine(
            time_limit=300,
            max_mcts_passes=25,
            generator_path=generator_path,
            cllp_path=cllp_path,
            cllp_num_beams=1,
            cllp_num_return_sequences=1,
            generator_num_beams=24,
            generator_num_subgoals=12,
            sort_subgoals_by=sort_subgoals_by,
            num_top_subgoals=6,
        )
        leela_log_dir = out_dir + "/leela"
        Path(leela_log_dir).mkdir(parents=True, exist_ok=True)
        self.engine_LEELA = PolicyChess(
            policy_checkpoint=None,
            log_dir=out_dir,
            debug_mode=True,
            replace_legall_move_with_random=False,
            do_sample=False,
            name="LeelaChessZero_POLICY",
            use_lczero_policy=True,
        )
        self.stockfish_bot = StockfishBotEngine(stockfish_depth=stockfish_bot_depth, stockfish_path=stockfish_path)

        self.players = {"w": self.engine_LEELA, "b": self.stockfish_bot}
        log_object("Players", f"White: {self.players['w'].name}, Black: {self.players['b'].name}")

    def execute(self):
        board = chess.Board()
        moves_counter = 0
        while not board.is_game_over():

            if board.turn == chess.WHITE:
                engine = self.players["w"]
                player = self.players["w"].name
            else:
                engine = self.players["b"]
                player = self.players["w"].name

            move = engine.propose_best_moves(board, 1)
            print(f"Move {moves_counter} by {player}: {move}")
            log_object("Move", f"Move {moves_counter} by {player}: {move}")
            board.push(chess.Move.from_uci(move))
            eval = self.stockfish.evaluate_immutable_board(ImmutableBoard.from_board(board))
            log_value("Stockfish evaluation", moves_counter, eval)
            fig = immutable_boards_to_img([ImmutableBoard.from_board(board)], [f"move {moves_counter}"])
            log_object("game", fig)
            moves_counter += 1

        log_object("Result", board.result())
