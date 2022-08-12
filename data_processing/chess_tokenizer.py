from collections import namedtuple

from chess import Move, PIECE_SYMBOLS

BoardState = namedtuple("BoardState", "board active_player castles")

PIECE_SYMBOL_TO_INT = {PIECE_SYMBOLS[i]: i for i in range(1, 7)}
INT_TO_PIECE_SYMBOL = {i: PIECE_SYMBOLS[i] for i in range(1, 7)}
TOKENIZED_BOARD_LENGTH = 73
NON_SPECIAL_TOKENS_START = 11


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
    non_special_vocab = pieces + squares + players + castlings
    special_vocab_to_tokens = {"<BOS>": 0, "<PAD>": 1, "<EOS>": 2, "<SEP>": 3}
    vocab_to_tokens = {
        symbol: i + NON_SPECIAL_TOKENS_START
        for i, symbol in enumerate(non_special_vocab)
    }
    vocab_to_tokens.update(special_vocab_to_tokens)
    tokens_to_vocab = {v: k for k, v in vocab_to_tokens.items()}


    @classmethod
    def encode_board(cls, board):
        board_state = board_to_board_state(board)
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


def board_to_board_state(board):
    return fen_to_board_state(board.fen())

