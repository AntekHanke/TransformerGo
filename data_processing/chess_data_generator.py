from dataclasses import dataclass

import chess.pgn
import random
import torch
from collections import namedtuple

from chess import Move, PIECE_SYMBOLS

BoardState = namedtuple("BoardState", "board active_player castles")
# ChessMove = namedtuple('ChessMove', 'from_square to_square promotion')
Transition = namedtuple("Transition", "board move")
OneGameData = namedtuple("OneGameData", "metadata, transitions")

PIECE_SYMBOL_TO_INT = {PIECE_SYMBOLS[i]: i for i in range(1, 7)}
INT_TO_PIECE_SYMBOL = {i: PIECE_SYMBOLS[i] for i in range(1, 7)}
TOKENIZED_BOARD_LENGTH = 73


class ChessTokenizer:
    pieces = [" ", "P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k", "/", "."]
    squares = list(range(0, 64))
    castlings = [
        "KQkq",
        "KQk",
        "KQ",
        "K",
        "Qkq",
        "Qk",
        "Q",
        "kq",
        "k",
        "q",
        "Kkq",
        "Kq",
        "Kk",
        "Qq",
        "KQq",
        "-",
    ]
    players = ["w", "b"]
    special_tokes = ["<SEP>", "<EOS>"]
    vocab = special_tokes + pieces + squares + players + castlings
    vocab_to_tokens = {symbol: i for i, symbol in enumerate(vocab)}
    tokens_to_vocab = {i: symbol for i, symbol in enumerate(vocab)}

    @classmethod
    def encode_board(cls, board_state):
        board_string = ""
        board_tokens = []
        for c in board_state.board:
            if c.isdigit() in list(range(1, 9)):
                board_string += "." * int(c)
            else:
                board_string += c

        for c in board_string:
            board_tokens.append(cls.vocab_to_tokens[c])

        board_tokens.append(cls.vocab_to_tokens[board_state.active_player])
        board_tokens.append(cls.vocab_to_tokens[board_state.castles])

        assert (
            len(board_tokens) == 73
        ), f"The number of tokens encoding the board must be 71, got len(board_tokens) = {len(board_tokens)}"
        return board_tokens

    @classmethod
    def decode_board(cls, board_tokens):
        board_string_with_dots = ""
        for i, token in enumerate(board_tokens):
            if i == 71 or i == 72:
                board_string_with_dots += " "
            board_string_with_dots += cls.tokens_to_vocab[token]

        board_string = ""
        dots_counter = 0

        for c in board_string_with_dots:
            if c == ".":
                dots_counter += 1
            else:
                if dots_counter > 0:
                    board_string += str(dots_counter)
                    dots_counter = 0
                board_string += c
        return board_string

    @classmethod
    def encode_move(cls, chess_move):
        move_tokens = [
            cls.vocab_to_tokens[chess_move.from_square],
            cls.vocab_to_tokens[chess_move.to_square],
        ]
        if chess_move.promotion is not None:
            move_tokens.append(
                cls.vocab_to_tokens[INT_TO_PIECE_SYMBOL[chess_move.promotion]]
            )
        else:
            move_tokens.append(cls.vocab_to_tokens["-"])

        move_tokens.append(cls.vocab_to_tokens["<EOS>"])
        return move_tokens

    @classmethod
    def decode_move(cls, move_tokens):
        promotion_str = cls.tokens_to_vocab[move_tokens[-1]]
        if promotion_str == "-":
            promotion = None
        else:
            promotion = PIECE_SYMBOL_TO_INT[promotion_str]
        return Move(
            cls.tokens_to_vocab[move_tokens[0]],
            cls.tokens_to_vocab[move_tokens[1]],
            promotion,
        )


def fen_to_board_state(fen_str):
    fen_components = fen_str.split()
    return BoardState(
        board=fen_components[0],
        active_player=fen_components[1],
        castles=fen_components[2],
    )


class ChessDataGenerator(torch.utils.data.Dataset):
    def __init__(self, pgn_file, p_sample=1.0, n_data=None, mode="train"):
        self.pgn_database = open(pgn_file, errors='ignore')
        self.p_sample = p_sample
        self.n_data = n_data
        self.mode = mode
        self.data_queue = {}
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
                transitions.append(Transition(fen_to_board_state(board.fen()), chess_move))
            except:
                break

        return OneGameData(chess_metadata, transitions)

    def create_data(self):
        n_iterations = 0
        while len(self.data_queue) < self.n_data:
            n_iterations += 1
            self.game_data_to_train_data(self.next_game_to_raw_data())
            if n_iterations % 1000 == 0:
                print(
                    f"Preparing dataset. Current len(self.data_queue) = {len(self.data_queue)} = {(10000*len(self.data_queue)/self.n_data)//100} %"
                )
        print(
            f"Final dataset {self.mode}. Current len(self.data_queue) = {len(self.data_queue)} = {(10000 * len(self.data_queue) / self.n_data) // 100} % \n \n"
        )

    def __getitem__(self, idx):
        # if idx not in self.data_queue:
        #     self.extend_data(idx)
        return self.data_queue[idx]

    # def extend_data(self, idx):
    #     num_trials = 0
    #     if idx not in self.data_queue:
    #         num_trials += 1
    #         if num_trials > self.length:
    #             raise ValueError(
    #                 f"Could not find new data from games, maybe there is too little games in the PGN dataset. len(self.data_queue = {len(self.data_queue)} idx = {idx}")
    #         self.game_data_to_train_data(self.next_game_to_raw_data())

    def game_data_to_train_data(self, game):
        raise NotImplementedError

    def __len__(self):
        return self.n_data


class ChessMovesDataGenerator(ChessDataGenerator):
    def game_data_to_train_data(self, game):
        for transition in game.transitions:
            if random.random() <= self.p_sample:
                self.data_queue[len(self.data_queue)] = {
                    "input_ids": ChessTokenizer.encode_board(transition.board),
                    "labels": ChessTokenizer.encode_move(transition.move),
                }


# x = ChessMovesDataGenerator("/home/tomek/Research/subgoal_chess_data/chess_micro_aa")
# x.__getitem__(3000)
# d = 4
