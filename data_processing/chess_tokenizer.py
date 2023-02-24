import random
from abc import abstractmethod
from typing import List, Union

from chess import Move

from configures.global_config import TOKENIZER, RANDOM_TOKENIZATION_ORDER
from data_structures.data_structures import ImmutableBoard

NON_SPECIAL_TOKENS_START = 11


class MoveDocodingException(Exception):
    pass


def is_promotion_possible(algebraic_move: str) -> bool:
    return (
        abs(int(algebraic_move[1]) - int(algebraic_move[3])) == 1
        and algebraic_move[3] in "18"
        and abs(ord(algebraic_move[0]) - ord(algebraic_move[2])) <= 1
    )


def padding(sequence: List, pad_value: int, final_len: int) -> List:
    assert len(sequence) <= final_len, "Sequence is longer than final_len"
    return sequence + [pad_value] * (final_len - len(sequence))


class ChessTokenizer:
    """Custom tokenizer for chess data."""

    pieces = [" ", "P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k", "/", "."]
    column_letters = ["a", "b", "c", "d", "e", "f", "g", "h"]
    letter_to_column = {letter: i for i, letter in enumerate(column_letters)}
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
    non_special_vocab = (
        pieces + integers + algebraic_fields + players + castlings + algebraic_moves + algebraic_promotions
    )
    special_vocab_to_tokens = {"<BOS>": 0, "<PAD>": 1, "<EOS>": 2, "<SEP>": 3}
    vocab_to_tokens = {symbol: i + NON_SPECIAL_TOKENS_START for i, symbol in enumerate(non_special_vocab)}
    vocab_to_tokens.update(special_vocab_to_tokens)
    tokens_to_vocab = {v: k for k, v in vocab_to_tokens.items()}

    def __new__(cls):
        if TOKENIZER == "board":
            self = ChessTokenizerBoard.__new__(ChessTokenizerBoard)
        elif TOKENIZER == "pieces":
            self = ChessTokenizerPiece.__new__(ChessTokenizerPiece)
        else:
            raise Exception(f"Tokenizer {TOKENIZER} not recognized. Should be either 'board' or 'pieces'.")
        return self

    @classmethod
    @abstractmethod
    def encode_immutable_board(cls, immutable_board: ImmutableBoard) -> List[int]:
        return cls.__new__(cls).encode_immutable_board(immutable_board)

    @classmethod
    @abstractmethod
    def decode_board(cls, board_tokens: List[int]) -> ImmutableBoard:
        return cls.__new__(cls).decode_board(board_tokens)

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

    @classmethod
    def decode_moves(cls, tokens):
        """Decode moves"""
        return [
            Move.from_uci(cls.tokens_to_vocab[token])
            for token in tokens
            if token not in cls.special_vocab_to_tokens.values()
        ]


class ChessTokenizerBoard(ChessTokenizer):
    TOKENIZED_BOARD_LENGTH = 76

    def __new__(cls):
        self = object.__new__(cls)
        return self

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


class ChessTokenizerPiece(ChessTokenizer):
    TOKENIZED_BOARD_LENGTH = 60

    def __new__(cls):
        self = object.__new__(cls)
        return self

    @classmethod
    def encode_immutable_board(cls, immutable_board: ImmutableBoard) -> List[int]:
        board_tokens = []
        board_row = 7
        board_column = 0
        piece_positions = {piece: [] for piece in cls.pieces[1:-2]}
        for c in immutable_board.board:
            if c.isdigit() in list(range(1, 9)):
                board_column += int(c)
            elif c in cls.pieces[1:-2]:
                piece_positions[c].append(cls.column_letters[board_column] + str(board_row + 1))
                board_column += 1
            if board_column == 8:
                board_row -= 1
                board_column = 0

        for piece in cls.pieces[1:-2]:
            if RANDOM_TOKENIZATION_ORDER:
                random.shuffle(piece_positions[piece])
            for position in piece_positions[piece]:
                board_tokens.append(cls.vocab_to_tokens[position])
            board_tokens.append(cls.vocab_to_tokens["<SEP>"])

        board_tokens.append(cls.vocab_to_tokens[immutable_board.active_player])
        board_tokens.append(cls.vocab_to_tokens[immutable_board.castles])
        board_tokens.append(cls.vocab_to_tokens[immutable_board.en_passant_target])
        halfmove_clock = min(int(immutable_board.halfmove_clock), 255)
        board_tokens.append(cls.vocab_to_tokens[str(halfmove_clock)])
        fullmove_clock = min(int(immutable_board.fullmove_clock), 255)
        board_tokens.append(cls.vocab_to_tokens[str(fullmove_clock)])

        assert len(board_tokens) <= cls.TOKENIZED_BOARD_LENGTH, (
            f"The number of tokens encoding the board must be less or equal than {cls.TOKENIZED_BOARD_LENGTH}, "
            f"got len(board_tokens) = {len(board_tokens)}"
        )
        board_tokens = padding(board_tokens, cls.vocab_to_tokens["<PAD>"], cls.TOKENIZED_BOARD_LENGTH)

        assert len(board_tokens) == cls.TOKENIZED_BOARD_LENGTH, (
            f"The number of tokens encoding the board must be {cls.TOKENIZED_BOARD_LENGTH}, "
            f"got len(board_tokens) = {len(board_tokens)}"
        )
        return board_tokens

    @classmethod
    def decode_board(cls, board_tokens: List[int]) -> ImmutableBoard:
        board_with_dots = [["." for j in range(8)] + ["/"] for j in range(8)]
        additional_data = ""
        piece_number = 0
        for token in board_tokens:
            vocab = cls.tokens_to_vocab[token]
            if piece_number < len(cls.pieces[1:-2]):
                if vocab == "<SEP>":
                    piece_number += 1
                else:
                    x = cls.letter_to_column[vocab[0]]
                    y = 8 - int(vocab[1])
                    board_with_dots[y][x] = cls.pieces[1:-2][piece_number]
            elif vocab != "<PAD>":
                additional_data += vocab + " "
            else:
                break
        flat_board_with_dots = [square for row in board_with_dots for square in row]
        additional_data = additional_data[:-1]

        board_string = ""
        dots_counter = 0

        for c in flat_board_with_dots:
            if c == ".":
                dots_counter += 1
            else:
                if dots_counter > 0:
                    board_string += str(dots_counter)
                    dots_counter = 0
                board_string += c
        board_string = board_string[:-1]
        board_string += " " + additional_data

        return ImmutableBoard(*board_string.split())


class ChessTokenizerFEN(ChessTokenizer):
    TOKENIZED_BOARD_LENGTH = 82

    def __new__(cls):
        self = object.__new__(cls)
        return self

    @classmethod
    def encode_immutable_board(cls, immutable_board: ImmutableBoard) -> List[int]:
        board_tokens = [cls.vocab_to_tokens[x] for x in immutable_board.board]
        board_tokens.append(cls.special_vocab_to_tokens["<SEP>"])
        board_tokens.append(cls.vocab_to_tokens[immutable_board.active_player])
        board_tokens.append(cls.special_vocab_to_tokens["<SEP>"])

        board_tokens.append(cls.vocab_to_tokens[immutable_board.castles])
        board_tokens.append(cls.special_vocab_to_tokens["<SEP>"])
        board_tokens.append(cls.vocab_to_tokens[immutable_board.en_passant_target])
        board_tokens.append(cls.special_vocab_to_tokens["<SEP>"])

        halfmove_clock = min(int(immutable_board.halfmove_clock), 255)
        board_tokens.append(cls.vocab_to_tokens[str(halfmove_clock)])
        board_tokens.append(cls.special_vocab_to_tokens["<SEP>"])

        fullmove_clock = min(int(immutable_board.fullmove_clock), 255)
        board_tokens.append(cls.vocab_to_tokens[str(fullmove_clock)])
        board_tokens.append(cls.special_vocab_to_tokens["<EOS>"])

        padding(board_tokens, cls.vocab_to_tokens["<PAD>"], cls.TOKENIZED_BOARD_LENGTH)

        return board_tokens

    @classmethod
    def decode_board(cls, board_tokens: List[int]) -> ImmutableBoard:
        decoded_board = [cls.tokens_to_vocab[token] for token in board_tokens]
        board_string = "".join(decoded_board)
        board_string = board_string.replace("<SEP>", " ").replace("<EOS>", "")
        return ImmutableBoard.from_fen_str(board_string)
