from typing import Dict, Any

import chess.pgn
import random
import torch
from collections import namedtuple

from data_processing.chess_tokenizer import ChessTokenizer
from data_structures.data_structures import ImmutableBoard, ChessMetadata, Transition, OneGameData
from data_processing.data_utils import get_split, immutable_boards_to_img
from metric_logging import log_value, log_object


# TODO: fill in fields
GameMetadata = namedtuple("GameMetadata", "game_id, winner, result")


class ChessFilter:
    """Filters games and transitions to use in training."""

    @staticmethod
    def use_game(game_metadata: ChessMetadata) -> bool:
        """Decides whether a game is useful."""
        raise NotImplementedError

    @staticmethod
    def use_transition(transition: Transition, one_game_data: OneGameData) -> bool:
        """Decides whether a transition is useful."""
        raise NotImplementedError


class NoFilter(ChessFilter):
    """Accepts every game and transition"""

    @staticmethod
    def use_game(game_metadata: ChessMetadata) -> bool:
        return True

    @staticmethod
    def use_transition(transition: Transition, one_game_data: OneGameData) -> bool:
        return True


class ResultFilter(ChessFilter):
    """Filters games that have a winner or looser. Filters transitions that were played by the winner or looser."""

    def __init__(self, winner_or_looser: str):
        assert winner_or_looser in ["winner", "loser"], "winner_or_looser must be 'winner' or 'loser'"
        self.winner_or_looser = winner_or_looser
        self.result_to_winner = {"1-0": "w", "0-1": "b"}

    @staticmethod
    def use_game(game_metadata: ChessMetadata) -> bool:
        return game_metadata.Result in ["1-0", "0-1"]

    def use_transition(self, transition: Transition, one_game_data: OneGameData) -> bool:
        if self.winner_or_looser == "winner":
            return transition.immutable_board.active_player == self.result_to_winner[one_game_data.metadata.Result]
        elif self.winner_or_looser == "loser":
            return transition.immutable_board.active_player != self.result_to_winner[one_game_data.metadata.Result]


class ChessDataset(torch.utils.data.Dataset):
    """Used by Pytorch DataLoader to get batches of data."""

    def __init__(self, data: Dict):
        self.data = data

    def __getitem__(self, idx: int):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


class ChessDataGenerator:
    """Reads PGN file and creates data."""

    def __init__(
        self,
        pgn_file: str,
        chess_filter: ChessFilter = None,
        p_sample: float = 1.0,
        n_data: int = None,
        train_eval_split: float = 0.95,
        log_samples_limit: int = None,
        p_log_sample: float = 0.01,
        only_eval: bool = False,
    ):
        self.pgn_database = open(pgn_file, errors="ignore")
        assert chess_filter is not None, "Chess filter must be specified"
        self.chess_filter = chess_filter
        self.p_sample = p_sample
        self.n_data = n_data
        self.train_eval_split = train_eval_split
        self.log_samples_limit = log_samples_limit
        self.p_log_sample = p_log_sample
        self.only_eval = only_eval

        self.data_queue = {}
        self.eval_data_queue = {}
        self.logged_samples = 0
        self.n_games = 0
        self.data_constructed = False

   def next_game_to_raw_data(self) -> Optional[OneGameData]:
        """
        Function takes the next game from the set of chess games, checks if it passes through the filter used and
        return information about game chass.
        :return: OneGameData contains inforamtion about chess game.
        """

        current_game: chess.pgn.Game = chess.pgn.read_game(self.pgn_database)

        if current_game is None:  # Condition is met if there are no more games in the dataset.
            return None
        else:
            chess_metadata: ChessMetadata = ChessMetadata(**current_game.headers)
            transitions: List[Transition] = []

            if self.chess_filter.use_game(chess_metadata):
                board = chess.Board()
                for move in enumerate(current_game.mainline_moves()):
                    _, chess_move = move
                    try:
                        transitions.append(Transition(ImmutableBoard.from_board(board), chess_move))
                        board.push(chess_move)
                    except:
                        break

            return OneGameData(chess_metadata, transitions)

    def create_data(self):
        n_iterations = 0
        while len(self.data_queue) + len(self.eval_data_queue) < self.n_data:
            n_iterations += 1
            train_eval = get_split(n_iterations, self.train_eval_split)
            if not self.only_eval or train_eval == "eval":
                current_dataset = self.select_dataset(train_eval)
                self.game_to_datapoints(self.next_game_to_raw_data(), current_dataset)
                self.n_games += 1
                self.log_progress(n_iterations)
        self.data_constructed = True

    def get_train_set_generator(self) -> ChessDataset:
        assert self.data_constructed, "Data not constructed, call .create_data() first"
        return ChessDataset(self.data_queue)

    def get_eval_set_generator(self) -> ChessDataset:
        assert self.data_constructed, "Data not constructed, call .create_data() first"
        return ChessDataset(self.eval_data_queue)

    def select_dataset(self, train_eval) -> Dict:
        if train_eval == "train":
            current_dataset = self.data_queue
        elif train_eval == "eval":
            current_dataset = self.eval_data_queue
        else:
            raise ValueError(f"Unknown train_eval value. Expected 'train' or 'eval', got {train_eval}")
        return current_dataset

    def game_to_datapoints(self, one_game_data: OneGameData, current_dataset: Dict):
        """Converts a game to datapoints added to the current_dataset (train or eval)."""
        raise NotImplementedError

    def log_sample(self, sample: Any, game_metadata: ChessMetadata):
        if self.log_samples_limit is not None and random.random() <= self.p_log_sample:
            if self.logged_samples < self.log_samples_limit:
                log_object("Data sample", self.sample_to_log_object(sample, game_metadata))
                self.logged_samples += 1

    def sample_to_log_object(self, sample: Any, game_metadata: ChessMetadata) -> Any:
        raise NotImplementedError

    def log_progress(self, n_iterations: int):
        if n_iterations % 1000 == 0:
            log_value("Train dataset points", n_iterations, len(self.data_queue))
            log_value("Eval dataset points", n_iterations, len(self.eval_data_queue))
            log_value("Dataset games", n_iterations, self.n_games)
            log_value(
                "Dataset size",
                n_iterations,
                len(self.data_queue) + len(self.eval_data_queue),
            )
            log_value(
                "Dataset progress",
                n_iterations,
                (len(self.data_queue) + len(self.eval_data_queue)) / self.n_data,
            )


class PolicyDataGenerator(ChessDataGenerator):
    def game_to_datapoints(self, one_game_data: OneGameData, current_dataset: Dict):
        for num, transition in enumerate(one_game_data.transitions):
            if random.random() <= self.p_sample and self.chess_filter.use_transition(transition, one_game_data):
                current_dataset[len(current_dataset)] = {
                    "input_ids": ChessTokenizer.encode_immutable_board(transition.immutable_board)
                    + [ChessTokenizer.vocab_to_tokens["<SEP>"]],
                    "labels": ChessTokenizer.encode_move(transition.move),
                }
                sample = {"input_board": transition.immutable_board, "move": transition.move.uci(), "num": num}
                self.log_sample(sample, one_game_data.metadata)

    def sample_to_log_object(self, sample: Any, metadata: ChessMetadata):
        return immutable_boards_to_img(
            [sample["input_board"]],
            [f"{sample['num']} : {sample['move']}, result: {metadata.Result}"],
        )


class ChessSubgoalDataGenerator(ChessDataGenerator):
    def __init__(self, k, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.k = k

    def game_to_datapoints(self, one_game_data: OneGameData, current_dataset: Dict):
        game_length = len(one_game_data.transitions)

        for num in range(game_length - self.k):

            input_board = one_game_data.transitions[num].immutable_board
            target_board_num = min(game_length - 1, num + self.k)
            target_board = one_game_data.transitions[target_board_num].immutable_board

            if random.random() <= self.p_sample:
                current_dataset[len(current_dataset)] = {
                    "input_ids": ChessTokenizer.encode_immutable_board(input_board)
                    + [ChessTokenizer.vocab_to_tokens["<SEP>"]],
                    "labels": ChessTokenizer.encode_immutable_board(target_board),
                }

            sample = {
                "input_board": input_board,
                "target_board": target_board,
                "num": num,
                "move": one_game_data.transitions[num].move.uci(),
            }
            self.log_sample(sample, one_game_data.metadata)

    def sample_to_log_object(self, sample: Dict, metadata: ChessMetadata):
        return immutable_boards_to_img(
            [sample["input_board"], sample["target_board"]],
            [f"{sample['num']} : {sample['move']}, res: {metadata.Result}", ""],
        )
