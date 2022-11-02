# #!/home/tomasz/anaconda3/envs/subgoal_search_chess/bin/python -u
#
# from typing import List
#
# import chess
# import chess.engine
#
# from chess_engines.bots.basic_chess_engines import PolicyChess
#
# # UCI commands
# UCI_COMMAND: str = "uci"
# UCI_OK_COMMAND: str = "uciok"
# UCI_IS_READY_COMMAND: str = "isready"
# UCI_READY_OK_COMMAND: str = "readyok"
# UCI_POSITION_COMMAND: str = "position"
# UCI_MOVES_COMMAND: str = "moves"
# UCI_GO_COMMAND: str = "go"
# UCI_QUIT_COMMAND: str = "quit"
#
#
# def get_move_list(s: str) -> str:
#     move_list: str = ""
#     pos: int = s.find(UCI_MOVES_COMMAND)
#     if pos >= 0:
#         move_list: str = s[(pos + len(UCI_MOVES_COMMAND)) :]
#     return move_list
#
#
# def curent_state(move_list: str) -> chess.Board:
#     board: chess.Board = chess.Board()
#     moves: List[str] = move_list.split(" ")
#     for m in moves:
#         if m != "":
#             board.push_uci(m)
#     return board
#
#
# def log(s: str, prefix: str = ">>> ") -> None:
#     f = open("chess_engines/bots/log.txt", "a")
#     f.write(prefix + s + "\n")
#     f.close()
#
#
# def output(s):
#     print(s, flush=True)
#     log(s, "")
#
#
# def main():
#     log("Starting chess engine")
#     engine = PolicyChess()
#     log(f"Engine type {type(engine)}")
#     log("Chess engine started")
#     # engine: RandomChessEngine = RandomChessEngine()
#     move_list: str = ""
#
#     while True:
#         s = input()
#         commands: List[str] = s.split(" ")
#         if len(commands) == 0:
#             continue
#
#         if commands[0] == UCI_COMMAND:
#             output("id name " + str(engine.name))
#             output("id authors Tomek and G")
#             output("uciok")
#
#         elif commands[0] == UCI_IS_READY_COMMAND:
#             output("readyok")
#
#         elif commands[0] == UCI_POSITION_COMMAND:
#             move_list = get_move_list(s)
#
#         elif commands[0] == UCI_GO_COMMAND:
#             baord: chess.Board = curent_state(move_list)
#             best_move = engine.policy(baord)
#             output("bestmove" + " " + best_move)
#             # output("info depth {} score cp {} time {} nodes {} pv {}".format(1, 1, 1, 1, 1))
#         elif commands[0] == UCI_QUIT_COMMAND:
#             break
#
#
# if __name__ == "__main__":
#     main()
