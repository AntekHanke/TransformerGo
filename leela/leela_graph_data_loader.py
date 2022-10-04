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


class LeelaGMLTree:
    def __init__(self, path_to_data: str, input_board: str, create_all_states: bool = False):
        self.path_to_data = path_to_data
        self.input_board = input_board
        self.create_all_states = create_all_states

        self.graph = self.to_graph()

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

    def to_graph(self) -> nx.classes.digraph.DiGraph:
        G: nx.classes.digraph.DiGraph = nx.readwrite.gml.read_gml(self.path_to_data, label="id")
        if self.create_all_states:
            for node in G.nodes:
                self.create_state(node)
        return G

    def create_state(self, node):
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
        all_k_successors =  list(nx.dfs_successors(self.graph, node, depth_limit=k+1))
        k_successors_dist = {succ_node: self.distance_to_predecessors(succ_node, node) for succ_node in all_k_successors}
        return k_successors_dist

    def get_parent(self, node):
        return list(self.graph.predecessors(node))[0]

    def distance_to_predecessors(self, node, predecessor):
        dist = 0
        if node != predecessor:
            parent = self.get_parent(node)
            dist += 1
            while parent != predecessor:
                parent = self.get_parent(parent)
                dist += 1
        return dist

        # return [self.distance(node, predecessor) for predecessor in predecessors]



    # def distance(self, node1, node2):
    #     return nx.shortest_path_length(self.graph, node1, node2)
