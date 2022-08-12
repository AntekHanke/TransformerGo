from collections import namedtuple

import chess

ImmutableBoardData = namedtuple("ImmutableBoard", "board active_player castles en_passant_target halfmove_clock fullmove_clock")

class ImmutableBoard(ImmutableBoardData):
    @classmethod
    def from_board(cls, board):
        fen_data = board.fen().split()
        return cls(*fen_data)

    def to_board(self):
        return chess.Board(fen=" ".join(self))

x = chess.Board()
y1 = ImmutableBoard.from_board(x)
x2 = y1.to_board()


for k in y1:
    print(k)

# print(y1)


# # print(x.fen())
# x.push(chess.Move.from_uci("e2e4"))
# x.push(chess.Move.from_uci("d7d5"))
# y2 = ImmutableBoard.from_board(x)
# print(y2)
#
# print(y1)

# print(x.fen())
