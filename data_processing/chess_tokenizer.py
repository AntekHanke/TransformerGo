from typing import List

from chess import Move, PIECE_SYMBOLS

from data_structures.data_structures import ImmutableBoard

PIECE_SYMBOL_TO_INT = {PIECE_SYMBOLS[i]: i for i in range(1, 7)}
INT_TO_PIECE_SYMBOL = {i: PIECE_SYMBOLS[i] for i in range(1, 7)}
TOKENIZED_BOARD_LENGTH = 73
NON_SPECIAL_TOKENS_START = 11


class MoveDocodingException(Exception):
    pass


class ChessTokenizer:
    """Custom tokenizer for chess data."""

    pieces = [" ", "P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k", "/", "."]
    integers = [str(i) for i in range(0, 256)]
    algebraic_fields = [f"{i}{j}" for i in ["a", "b", "c", "d", "e", "f", "g", "h"] for j in range(1, 9)]

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
    non_special_vocab = pieces + integers + algebraic_fields + players + castlings
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
        move_tokens = [
            cls.vocab_to_tokens[str(chess_move.from_square)],
            cls.vocab_to_tokens[str(chess_move.to_square)],
        ]
        if chess_move.promotion is not None:
            move_tokens.append(cls.vocab_to_tokens[INT_TO_PIECE_SYMBOL[chess_move.promotion]])
        else:
            move_tokens.append(cls.vocab_to_tokens["-"])
        return move_tokens

    @classmethod
    def encode_leela_move(cls, chess_move_as_string: str) -> List[int]:
        move_tokens = [
            cls.vocab_to_tokens[chess_move_as_string[0:2]],
            cls.vocab_to_tokens[chess_move_as_string[2:4]],
        ]
        if len(chess_move_as_string) == 5:
            move_tokens.append(cls.vocab_to_tokens[chess_move_as_string[4]])
        else:
            move_tokens.append(cls.vocab_to_tokens["-"])
        return move_tokens

    @classmethod
    def decode_move(cls, output_tokens: List[int]) -> Move:
        output_tokens = [
            token for token in output_tokens if token not in ChessTokenizer.special_vocab_to_tokens.values()
        ]
        promotion_str = cls.tokens_to_vocab[output_tokens[2]]
        if promotion_str == "-":
            promotion = None
        else:
            promotion = PIECE_SYMBOL_TO_INT[promotion_str]
        return Move(
            int(cls.tokens_to_vocab[output_tokens[0]]),
            int(cls.tokens_to_vocab[output_tokens[1]]),
            promotion,
        )

    @classmethod
    def decode(cls, tokens):
        """General decode method"""
        return [cls.tokens_to_vocab[token] for token in tokens]

    @classmethod
    def decode_leela_moves(cls, output_tokens):
        """Decode Leela moves"""
        decoded_tokens = "".join(cls.decode(output_tokens))

    # @classmethod
    # def decode_many_moves(cls, output_tokens: List[int]) -> List[Move]:
    #     output_tokens = [x for x in output_tokens if x != ChessTokenizer.special_vocab_to_tokens["<PAD>"]]
    #     moves = []
    #     tokens_to_decode = []
    #     for token in output_tokens:
    #         if token != cls.special_vocab_to_tokens["<SEP>"]:
    #             tokens_to_decode.append(token)
    #         else:
    #             moves.append(cls.decode_move(tokens_to_decode))
    #             tokens_to_decode = []
    #     return moves
