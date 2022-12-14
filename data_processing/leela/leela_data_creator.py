from typing import List, Optional, TextIO
import chess.pgn
import numpy as np
import shutil
import os
import time
import chess.engine
from metric_logging import log_value


class LeelaTreesGenerator:
    """
    exaple of leela parmas
    ['/home/gracjan/lc0_for_lct/build/release/lc0',
     '--weights=/home/gracjan/lc0_for_lct/build/release744204.pb.gz',
     '--smart-pruning-factor=0.0',
     '--threads=1',
     '--minibatch-size=1',
     '--max-collision-events=1',
     '--max-collision-visits=1',
     '--cpuct=1.0'
     ]
    """

    def __init__(
            self,
            path_to_chess_dataset: str,
            leela_parms: List[str],
            number_of_searching_nodes: int = 1000
    ) -> None:
        self.path_to_chess_dataset = path_to_chess_dataset
        self.leela_parms = leela_parms
        self.number_of_searching_nodes = number_of_searching_nodes

    def chess_data_generator(self):
        with open(self.path_to_chess_dataset, 'r') as data:
            chess_database: TextIO = data

            while True:
                game_from_database: Optional[chess.pgn.Game] = chess.pgn.read_game(chess_database)
                if game_from_database is None:
                    break
                elif len(list(game_from_database.mainline_moves())) == 0:
                    continue
                else:
                    yield game_from_database

    @staticmethod
    def random_chess_board(game: chess.pgn.Game) -> str:
        moves: List[chess.Move] = list(game.mainline_moves())
        board: chess.Board = chess.Board()

        while True:
            stop_playing: int = np.random.randint(0, len(moves))
            for n, move in enumerate(moves):
                if n == stop_playing:
                    break
                else:
                    board.push(move)

            if not board.is_checkmate():
                break
            else:
                continue

        return board.fen()

    def play(self, input_board: str) -> None:
        board: chess.Board = chess.Board()
        board.set_fen(input_board)

        print("Playing game with arguments:", self.leela_parms)
        lc0: chess.engine.SimpleEngine = chess.engine.SimpleEngine.popen_uci(self.leela_parms)

        print("starting search, nodes= ", str(self.number_of_searching_nodes))
        start: float = time.time()
        lc0.play(board, chess.engine.Limit(nodes=self.number_of_searching_nodes))
        print("search completed in time: ", time.time() - start)
        lc0.quit()
        return None

    def generate_and_save_tress_by_leela(self) -> None:
        n_board: int = 0
        with open('leela/trees_of_leela/all_trees.txt', 'w') as leela_tress:

            for game in self.chess_data_generator():
                board: str = self.random_chess_board(game)
                self.play(input_board=board)

                shutil.move('tree.gml',
                            'leela/trees_of_leela/tree.gml')

                with open('leela/trees_of_leela/tree.gml', 'r') as curent_tree:
                    leela_tress.write(board)
                    leela_tress.write('\n')
                    for line in curent_tree:
                        leela_tress.write(line)
                    leela_tress.write('\n')
                n_board += 1
                log_value('evaluated board', n_board, n_board)
                os.remove('leela/trees_of_leela/tree.gml')