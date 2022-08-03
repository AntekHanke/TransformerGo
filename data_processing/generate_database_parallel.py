import os
from collections import namedtuple

import chess.pgn
import dill as dill
from joblib import Parallel, delayed

class SinglePGNFileReader():
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

def iterate_one_file(single_pgn_file_reader):
    return single_pgn_file_reader.next_game_to_data()

def iterate_one_file_n_times(single_pgn_file_reader, n):
    result = []
    succeed,failed = 0, 0
    for _ in range(n):
        try:
            result.append(iterate_one_file(single_pgn_file_reader))
            succeed += 1
        except:
            failed += 1

    print(f'len result = {len(result)}')
    return result, succeed, failed


def generate_raw_database(data_file):
    all_files = os.listdir(data_dir)
    stats = {'succeed': 0, 'failed': 0}
    # correct_file_names = [name for name in all_files if name.startswith(data_prefix)][:2]
    # file_readers = [SinglePGNFileReader(os.path.join(data_dir, name)) for name in correct_file_names]
    n_jobs = len(correct_file_names)
    print(correct_file_names)
    database = []
    n_iterations = 50
    # results = Parallel(n_jobs=n_jobs, backend='threading')(delayed(iterate_one_file_n_times)(file_reader, n_iterations) for file_reader in file_readers)
    results = iterate_one_file_n_times()
    for result in results:
        data, succeed, failed = result
        database.extend(data)
        stats['succeed'] += succeed
        stats['failed'] += failed
    return database, stats

import pickle

# x = SinglePGNFileReader('/home/tomek/Research/subgoal_search_chess/assets/cas_small.pgn')

database, stats = generate_raw_database('/home/tomek/Research/subgoal_chess_data', 'chess_micro')
#
print(stats)
# print(database)
print(len(database))



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

