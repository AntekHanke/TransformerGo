from typing import List, ByteString, Tuple, Dict

import chess
import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout


def get_moves(graph: nx.classes.digraph.DiGraph, n: int) -> List[str]:
    moves = []
    move = graph.nodes[n]["move"]
    while move != "":
        moves.append(move)
        n = list(graph.predecessors(n))[0]
        move = graph.nodes[n]["move"]
    return moves[::-1]


def current_position(input_board: str, moves: List[str]) -> str:
    board = chess.Board(input_board)
    for move in moves:
        move = chess.Move.from_uci(move)
        board.push(move)
    return board.fen()


class LeelaGMLDataLoader:
    def __init__(self, path_to_data: str):
        self.path_to_data = path_to_data

    def data_trees_generator(self) -> Tuple[str, List[ByteString]]:
        with open(self.path_to_data, "rb") as dataset:
            while True:
                graph: List[ByteString] = []
                input_board: str = dataset.readline().decode("UTF-8")[:-1]
                for line in dataset:
                    if line == b"\n":
                        break
                    graph.append(line)
                yield input_board, graph

    @staticmethod
    def to_graph(graph: List[ByteString]) -> nx.classes.digraph.DiGraph:
        G: nx.classes.digraph.DiGraph = nx.readwrite.gml.read_gml(graph, label="id")
        return G

    @staticmethod
    def to_states_graph(
        input_board: str, graph: nx.classes.digraph.DiGraph
    ) -> nx.classes.digraph.DiGraph:
        board = input_board
        lables: Dict[int, str] = {
            n: current_position(board, get_moves(graph, n)) for n in graph
        }
        return nx.relabel_nodes(graph, lables, copy=True)

    @staticmethod
    def visualize_states_graph(graph: nx.classes.digraph.DiGraph) -> None:
        pos = graphviz_layout(graph, prog="dot")
        nx.draw(graph, pos=pos, with_labels=True)
        plt.show()