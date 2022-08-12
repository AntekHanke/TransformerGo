# from collections import namedtuple
#
# import chess
#
# from data_processing.chess_tokenizer import ChessTokenizer
#
# ImmutableBoardData = namedtuple("ImmutableBoard", "board active_player castles en_passant_target halfmove_clock fullmove_clock")
#
# class ImmutableBoard(ImmutableBoardData):
#     @classmethod
#     def from_board(cls, board):
#         fen_data = board.fen().split()
#         return cls(*fen_data)
#
#     def to_board(self):
#         return chess.Board(fen=" ".join(self))
#
# x = chess.Board()
# y1 = ImmutableBoard.from_board(x)
#
# eim = ChessTokenizer.encode_immutable_board(y1)
# dim = ChessTokenizer.decode_board(eim)
#
# new_board = dim.to_board()
#
# print(new_board)