import os
from collections import namedtuple

import chess.pgn
import tqdm as tqdm


class SinglePGNFileToData():
    def __init__(self, pgn_file):
        self.pgn_database = open(pgn_file)

    def next_game_to_data(self):
        self.current_game = chess.pgn.read_game(self.pgn_database)
        chess_metadata = namedtuple('chess_metadata', self.current_game.headers.keys())(*self.current_game.headers.values())
        board_list = []
        board = chess.Board()
        for moves in enumerate(self.current_game.mainline_moves()):
            board.push(moves[1])
            fen = board.fen()
            board_list.append(fen)

        return (chess_metadata, board_list)

    def get_data_batch(self, batch_size):
        database = []
        stats = {'succeed': 0, 'failed': 0}
        for i in tqdm.tqdm((range(batch_size))):
            try:
                new_data = self.next_game_to_data()
                database.append(new_data)
                stats['succeed'] += 1
            except:
                stats['failed'] += 1

        return database, stats

    def data_for_training(self):


x = SinglePGNFileToData('/home/tomek/Research/subgoal_chess_data/chess_data_aa')
data, stats = x.get_data_batch(1000)


# chess_micro_aa
# chess_micro_ab
# chess_micro_ac
#
# x = SinglePGNFileReader("/home/tomek/Research/subgoal_search_chess/assets/cas_small.pgn")
#
# print(iterate_one_file(x))
# print(iterate_one_file(x))
# print(iterate_one_file(x))
#

