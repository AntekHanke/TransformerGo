from typing import List, ByteString, Tuple, Dict, Iterable

import chess
import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout

from data_structures.data_structures import LeelaNodeData, LeelaSubgoal


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
                self.get_node_data(node)
        return G

    def get_node_data(self, node) -> LeelaNodeData:
        if "data" in self.graph.nodes[node]:
            return self.graph.nodes[node]["data"]
        else:
            moves_from_root = get_moves(self.graph, node)
            state = current_position(self.input_board, moves_from_root)
            print(self.graph.nodes[node])
            node_info = self.graph.nodes[node]
            data = LeelaNodeData(
                node,
                state,
                moves_from_root,
                len(moves_from_root),
                node_info["N"],
                node_info["Q"],
                node_info["D"],
                node_info["M"],
                node_info["P"],
            )
            self.graph.nodes[node]["data"] = data
            return data

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
        k_successors_idx = list(nx.dfs_successors(self.graph, node, depth_limit=k + 1))
        k_succesors = []
        for idx in k_successors_idx:
            moves_from_input = [x for x in get_moves(self.graph, idx) if x not in get_moves(self.graph, node)]
            succ = LeelaSubgoal(
                input_board=self.get_node_data(node).state,
                target_board=self.get_node_data(idx).state,
                dist_from_input=self.distance_to_predecessors(idx, node),
                input_level=self.get_node_data(node).level,
                moves=moves_from_input,
                N=self.get_node_data(idx).N,
                Q=self.get_node_data(idx).Q,
                D=self.get_node_data(idx).D,
                M=self.get_node_data(idx).M,
                P=self.get_node_data(idx).P,
            )
            k_succesors.append(succ)
        return k_succesors

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
