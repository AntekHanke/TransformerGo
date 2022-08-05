from dataclasses import dataclass

import chess.pgn
from collections import namedtuple

BoardState = namedtuple('BoardState', 'board active_player castles')
ChessMove = namedtuple('ChessMove', 'from_square to_square promotion')
OneGameData = namedtuple('OneGameData', 'metadata, boards_list, chess_moves_list')

class BoardState:
    board = None
    active_player = None
    castles = None

    def to_img(self):
        pass

    def show_img(self):

def fen_to_board_state(fen_str):
    fen_components = fen_str.split()
    return BoardState(board = fen_components[0], active_player=fen_components[1], castles=fen_components[2])

class ChessDataGenerator:
    def __init__(self, pgn_file):
        self.pgn_database = open(pgn_file)

    def next_game_to_raw_data(self):
        self.current_game = chess.pgn.read_game(self.pgn_database)
        self.current_game.mainline_moves()

        chess_metadata = namedtuple('chess_metadata', self.current_game.headers.keys())(*self.current_game.headers.values())
        boards_list = []
        chess_moves_list = []


        board = chess.Board()
        for move in enumerate(self.current_game.mainline_moves()):
            _, chess_move = move
            board.push(chess_move)
            chess_moves_list.append(ChessMove(chess_move.from_square, chess_move.to_square, chess_move.promotion))
            if chess_move.promotion is not None:
                print(ChessMove(chess_move.from_square, chess_move.to_square, chess_move.promotion))
            fen_str = board.fen()
            boards_list.append(fen_to_board_state(fen_str))

        return OneGameData(chess_metadata, boards_list, chess_moves_list)

# x = ChessDataGenerator("/home/tomek/Research/subgoal_chess_data/chess_micro_aa")
# for _ in range(20):
#     data = x.next_game_to_raw_data()
#     s = 4