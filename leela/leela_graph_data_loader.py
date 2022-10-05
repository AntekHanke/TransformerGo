from typing import List, ByteString, Tuple, Dict, Iterable

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


class LeelaGMLTree:
    def __init__(self, lines, input_board: str, create_all_states: bool = False):
        self.lines = lines
        self.input_board = input_board
        self.create_all_states = create_all_states

        self.graph = self.to_graph()

    def to_graph(self) -> nx.classes.digraph.DiGraph:
        G: nx.classes.digraph.DiGraph = nx.readwrite.gml.parse_gml(self.lines, label="id")
        if self.create_all_states:
            for node in G.nodes:
                self.create_state(node)
        return G

    def create_state(self, node) -> str:
        if "state" not in self.graph.nodes[node]:
            return self.graph.nodes[node]["state"]
        else:
            position = current_position(self.input_board, get_moves(self.graph, node))
            self.graph.nodes[node]["state"] = position
            return position

    def visualize_states_graph(self) -> None:
        fig, ax = plt.subplots(figsize=(60, 60))
        pos = graphviz_layout(self.graph, prog="dot")
        nx.draw(
            self.graph,
            pos=pos,
            ax=ax,
            with_labels=True,
        )
        plt.show()

    def k_successors(self, node, k):
        all_k_successors = list(nx.dfs_successors(self.graph, node, depth_limit=k + 1))
        for node in all_k_successors:
            self.create_state(node)
        k_successors_data = {
            succ_node: {
                "dist_from_input": self.distance_to_predecessors(succ_node, node),
                "state": self.graph.nodes[succ_node]["state"],
                "N": self.graph.nodes[succ_node]["N"],
                "Q": self.graph.nodes[succ_node]["Q"],
                "D": self.graph.nodes[succ_node]["D"],
                "M": self.graph.nodes[succ_node]["M"],
                "P": self.graph.nodes[succ_node]["P"],
            }
            for succ_node in all_k_successors
        }
        return k_successors_data

    def get_parent(self, node):
        return list(self.graph.predecessors(node))[0]

    def distance_to_predecessors(self, node, predecessor) -> int:
        dist = 0
        if node != predecessor:
            parent = self.get_parent(node)
            dist += 1
            while parent != predecessor:
                parent = self.get_parent(parent)
                dist += 1
        return dist


class EndOfGMLFile(Exception):
    pass

def data_trees_generator(data_path: str, create_all_states: bool = False) -> Iterable[LeelaGMLTree]:
    with open(data_path, "rb") as dataset:
        try:
            while True:
                graph_lines: List[str] = []
                input_board: str = dataset.readline().decode("UTF-8")[:-1]

                for line in dataset:
                    if not line:
                        raise EndOfGMLFile
                    line = line.decode("ascii")
                    if line == "\n":
                        break
                    graph_lines.append(line)
                try:
                    yield LeelaGMLTree(graph_lines, input_board, create_all_states)
                except nx.exception.NetworkXError:
                    raise EndOfGMLFile
        except EndOfGMLFile:
            print("End of GML file")