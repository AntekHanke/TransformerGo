import os.path
from typing import List

import chess
import chess.engine

from chess_engines.bots.basic_chess_engines import PolicyChess, ChessEngine
from metric_logging import turn_off_loggers

# UCI commands
UCI_COMMAND: str = "uci"
UCI_OK_COMMAND: str = "uciok"
UCI_IS_READY_COMMAND: str = "isready"
UCI_READY_OK_COMMAND: str = "readyok"
UCI_POSITION_COMMAND: str = "position"
UCI_MOVES_COMMAND: str = "moves"
UCI_GO_COMMAND: str = "go"
UCI_QUIT_COMMAND: str = "quit"
UCI_NEW_GAME: str = "ucinewgame"


def get_move_list(command_with_move_list: str) -> str:
    move_list: str = ""
    pos: int = command_with_move_list.find(UCI_MOVES_COMMAND)
    if pos >= 0:
        move_list: str = command_with_move_list[(pos + len(UCI_MOVES_COMMAND)):]
    return move_list


def move_list_from_str(moves_list_str: str) -> List[chess.Move]:
    move_list: List[str] = moves_list_str.split(" ")
    return [chess.Move.from_uci(move) for move in move_list if move != ""]


def curent_state(move_list: str) -> chess.Board:
    board: chess.Board = chess.Board()
    moves: List[str] = move_list.split(" ")
    for m in moves:
        if m != "":
            board.push_uci(m)
    return board


def log(s: str, prefix: str = ">>> ") -> None:
    f = open(os.path.join(os.environ["SUBGOAL_PROJECT_ROOT"], "chess_engines/bots/log.txt"), "a")
    f.write(prefix + s + "\n")
    f.close()


def output(s):
    print(s, flush=True)
    log(s, "")


def main_uci_loop(engine: ChessEngine):
    turn_off_loggers()
    log("Starting chess engine")
    log(f"Engine type {type(engine)}")
    log("Chess engine started")
    engine.new_game()

    move_list: str = ""

    while True:
        s = input()
        commands: List[str] = s.split(" ")
        log(f"Received command {s}")

        if len(commands) == 0:
            continue

        if commands[0] == UCI_COMMAND:
            output("id name " + str(engine.name))
            output("id authors Tomek, Gracjan and Gosia")
            output(UCI_OK_COMMAND)

        elif commands[0] == UCI_IS_READY_COMMAND:
            output(UCI_READY_OK_COMMAND)

        elif commands[0] == UCI_POSITION_COMMAND:
            move_list = get_move_list(s)

        elif commands[0] == UCI_GO_COMMAND:
            board: chess.Board = curent_state(move_list)
            history: List[chess.Move] = move_list_from_str(move_list)
            best_move = engine.propose_best_moves(current_state=board, number_of_moves=8, history=history)
            output("bestmove" + " " + best_move)

        elif commands[0] == UCI_QUIT_COMMAND:
            break
