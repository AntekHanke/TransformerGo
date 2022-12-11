from typing import List, Union

from chess import Move, PIECE_SYMBOLS

from data_structures.data_structures import ImmutableBoard

PIECE_SYMBOL_TO_INT = {PIECE_SYMBOLS[i]: i for i in range(1, 7)}
INT_TO_PIECE_SYMBOL = {i: PIECE_SYMBOLS[i] for i in range(1, 7)}
TOKENIZED_BOARD_LENGTH = 73
NON_SPECIAL_TOKENS_START = 11


class MoveDocodingException(Exception):
    pass


def is_promotion_possible(algebraic_move: str) -> bool:
    return (
        abs(int(algebraic_move[1]) - int(algebraic_move[3])) == 1
        and algebraic_move[3] in "18"
        and abs(ord(algebraic_move[0]) - ord(algebraic_move[2])) <= 1
    )


class ChessTokenizer:
    """Custom tokenizer for chess data."""

    pieces = [" ", "P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k", "/", "."]
    integers = [str(i) for i in range(0, 256)]
    algebraic_fields = [f"{i}{j}" for i in ["a", "b", "c", "d", "e", "f", "g", "h"] for j in range(1, 9)]
    algebraic_moves = []
    for start in algebraic_fields:
        for end in algebraic_fields:
            if start != end:
                algebraic_moves.append(f"{start}{end}")

    algebraic_promotions = []
    for move in algebraic_moves:
        if is_promotion_possible(move):
            for promotion in ["q", "r", "b", "n"]:
                algebraic_promotions.append(f"{move}{promotion}")

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
    non_special_vocab = pieces + integers + algebraic_fields + players + castlings + algebraic_moves + algebraic_promotions
    special_vocab_to_tokens = {"<BOS>": 0, "<PAD>": 1, "<EOS>": 2, "<SEP>": 3}
    vocab_to_tokens = {symbol: i + NON_SPECIAL_TOKENS_START for i, symbol in enumerate(non_special_vocab)}
    vocab_to_tokens.update(special_vocab_to_tokens)
    tokens_to_vocab = {v: k for k, v in vocab_to_tokens.items()}

    TOKENIZED_BOARD_LENGTH = 76

    @classmethod
    def encode_immutable_board(cls, immutable_board: ImmutableBoard) -> List[int]:
        board_string = ""
        board_tokens = []
        for c in immutable_board.board:
            if c.isdigit() in list(range(1, 9)):
                board_string += "." * int(c)
            else:
                board_string += c

        for c in board_string:
            board_tokens.append(cls.vocab_to_tokens[c])

        board_tokens.append(cls.vocab_to_tokens[immutable_board.active_player])
        board_tokens.append(cls.vocab_to_tokens[immutable_board.castles])
        board_tokens.append(cls.vocab_to_tokens[immutable_board.en_passant_target])
        halfmove_clock = min(int(immutable_board.halfmove_clock), 255)
        board_tokens.append(cls.vocab_to_tokens[str(halfmove_clock)])
        fullmove_clock = min(int(immutable_board.fullmove_clock), 255)
        board_tokens.append(cls.vocab_to_tokens[str(fullmove_clock)])

        assert (
            len(board_tokens) == cls.TOKENIZED_BOARD_LENGTH
        ), f"The number of tokens encoding the board must be {cls.TOKENIZED_BOARD_LENGTH}, got len(board_tokens) = {len(board_tokens)}"
        return board_tokens

    @classmethod
    def decode_board(cls, board_tokens: List[int]) -> ImmutableBoard:
        board_string_with_dots = ""
        board_tokens = [token for token in board_tokens if token not in ChessTokenizer.special_vocab_to_tokens.values()]
        board_tokens = board_tokens[: cls.TOKENIZED_BOARD_LENGTH]
        for i, token in enumerate(board_tokens):
            if i >= 71:
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
        return ImmutableBoard(*board_string.split())

    @classmethod
    def encode_move(cls, chess_move: Move) -> List[int]:
        return [cls.vocab_to_tokens[chess_move.uci()]]

    @classmethod
    def decode_move(cls, move_token: List[int]) -> Move:
        return Move.from_uci(cls.tokens_to_vocab[move_token[0]])

    @classmethod
    def encode(cls, str_or_str_list: Union[List[str], str]) -> List[int]:
        if isinstance(str_or_str_list, list):
            return [cls.vocab_to_tokens[s] for s in str_or_str_list]
        elif isinstance(str_or_str_list, str):
            return [cls.vocab_to_tokens[str_or_str_list]]
        else:
            raise ValueError("str_or_str_list must be a list of strings or a string")

    @classmethod
    def decode(cls, tokens):
        """General decode method"""
        return [cls.tokens_to_vocab[token] for token in tokens]
