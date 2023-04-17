import random

from chess_engines.third_party.stockfish import StockfishEngine
from data_processing.chess_data_generator import NoFilter, ChessSubgoalGamesDataGenerator
from data_processing.chess_tokenizer import ChessTokenizer
from utils.data_utils import immutable_boards_to_img
from jobs.core import Job

from metric_logging import log_object
from subgoal_generator.subgoal_generator import ChessSubgoalGenerator


class EvaluateGenerator(Job):
    def __init__(
        self,
        k: int,
        n_subgoals: int,
        subgoal_generator: ChessSubgoalGenerator,
        pgn_file: str = None,
        n_eval_datapoints: int = 1000,
        n_log_samples: int = 100,
    ):

        self.k = k
        self.n_subgoals = n_subgoals
        self.subgoal_generator = subgoal_generator
        self.chess_database = ChessSubgoalGamesDataGenerator(
            self.k, pgn_file, NoFilter(), p_sample=0.5, n_data=100 * n_eval_datapoints, only_eval=True
        )
        self.n_eval_datapoints = n_eval_datapoints
        self.n_log_samples = n_log_samples
        self.stockfish = StockfishEngine()

    def execute(self):
        self.chess_database.create_data()
        eval_data = self.chess_database.get_eval_set_generator()
        eval_data_len = len(eval_data)
        idx_to_eval = random.choices(list(range(eval_data_len)), k=self.n_eval_datapoints)
        for idx in idx_to_eval:
            input_board = ChessTokenizer.decode_board(eval_data[idx]["input_ids"])
            input_eval = self.stockfish.evaluate_immutable_board(input_board)
            data_target = ChessTokenizer.decode_board(eval_data[idx]["labels"])
            target_eval = self.stockfish.evaluate_immutable_board(data_target)
            subgoals = self.subgoal_generator.generate_subgoals(input_board=input_board,
                                                                generator_num_beams=16,
                                                                generator_num_subgoals=self.n_subgoals,
                                                                subgoal_distance_k=3)
            subgoals = [s for s in subgoals if s.board != input_board.board]
            subgoals_values = [self.stockfish.evaluate_immutable_board(subgoal) for subgoal in subgoals]

            fig = immutable_boards_to_img(
                [input_board, data_target, *subgoals],
                [f"input v={input_eval} cp  active = {input_board.active_player}"]
                + [f"subgoal v={v} cp" for v in subgoals_values]
                + [f"target v={target_eval} cp"],
            )
            log_object("Subgoals", fig)
