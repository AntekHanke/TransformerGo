import pickle

import chess.pgn
import random
import torch
from collections import namedtuple

from data_processing.chess_tokenizer import (
    fen_to_immutable_board,
    ChessTokenizer,
    board_to_immutable_board,
)
from data_processing.data_utils import get_split, boards_to_img
from metric_logging import log_value, log_object

Transition = namedtuple("Transition", "board move")
OneGameData = namedtuple("OneGameData", "metadata, transitions")


class ChessDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


class ChessDataGenerator:
    def __init__(self, pgn_file, p_sample=1.0, n_data=None, train_eval_split=0.95, log_samples_limit=None):
        self.pgn_database = open(pgn_file, errors="ignore")
        self.p_sample = p_sample
        self.n_data = n_data
        self.train_eval_split = train_eval_split
        self.log_samples_limit = log_samples_limit
        self.data_queue = {}
        self.eval_data_queue = {}
        self.logged_samples = 0


        self.n_games = 0
        self.create_data()

    def next_game_to_raw_data(self):
        self.current_game = chess.pgn.read_game(self.pgn_database)
        self.current_game.mainline_moves()

        chess_metadata = namedtuple("chess_metadata", self.current_game.headers.keys())(
            *self.current_game.headers.values()
        )
        transitions = []
        board = chess.Board()

        for move in enumerate(self.current_game.mainline_moves()):
            _, chess_move = move
            try:
                board.push(chess_move)
                transitions.append(Transition(board_to_immutable_board(board), chess_move))
            except:
                break

        return OneGameData(chess_metadata, transitions)

    def create_data(self):
        n_iterations = 0
        while len(self.data_queue) + len(self.eval_data_queue) < self.n_data:
            n_iterations += 1
            train_eval = get_split(n_iterations, self.train_eval_split)
            self.game_to_dataset(self.next_game_to_raw_data(), train_eval)
            self.n_games += 1
            self.log_progress(n_iterations)


    def log_progress(self, n_iterations):
        if n_iterations % 1000 == 0:
            log_value("Train dataset points", n_iterations, len(self.data_queue))
            log_value(
                "Eval dataset points", n_iterations, len(self.eval_data_queue)
            )
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


    def get_train_set_generator(self):
        return ChessDataset(self.data_queue)

    def get_eval_set_generator(self):
        return ChessDataset(self.eval_data_queue)

    def game_to_dataset(self, game, train_eval):
        raise NotImplementedError


class PolicyDataGenerator(ChessDataGenerator):
    def game_to_dataset(self, game, train_eval):
        if train_eval == "train":
            current_dataset = self.data_queue
        elif train_eval == "eval":
            current_dataset = self.eval_data_queue
        else:
            raise ValueError(
                f"Uknown train_eval value. Expected 'train' or 'eval', got {train_eval}"
            )
        for transition in game.transitions:
            if random.random() <= self.p_sample:
                current_dataset[len(current_dataset)] = {
                    "input_ids": ChessTokenizer.encode_immutable_board(transition.board),
                    "labels": ChessTokenizer.encode_move(transition.move),
                }
                if self.log_samples_limit is not None:
                    if self.logged_samples < self.log_samples_limit:
                        log_object('Data sample', boards_to_img([transition.board], str(transition.move)))
                        self.logged_samples += 1

class ChessSubgoalDataGenerator(ChessDataGenerator):
    def game_to_dataset(self, game, train_eval):
        if train_eval == "train":
            current_dataset = self.data_queue
        elif train_eval == "eval":
            current_dataset = self.eval_data_queue
        else:
            raise ValueError(
                f"Uknown train_eval value. Expected 'train' or 'eval', got {train_eval}"
            )
        for transition in game.transitions:
            if random.random() <= self.p_sample:
                current_dataset[len(self.data_queue)] = {
                    "input_ids": ChessTokenizer.encode_immutable_board(transition.board),
                    "labels": ChessTokenizer.encode_move(transition.move),
                }
