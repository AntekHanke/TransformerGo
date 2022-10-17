from collections import defaultdict
from typing import List, ByteString, Tuple, Dict, Iterable, Iterator

import chess
import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout

from data_structures.data_structures import ImmutableBoard


class EndOfGMLFile(Exception):
    pass


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


class LeelaNodeData:
    def __init__(self, id, params, moves_from_root, input_board_fen):
        self.id = id
        self.params = params
        self.moves_from_root = moves_from_root
        self.input_board_fen = input_board_fen

    def get_numerical_param(self, param_name):
        assert param_name in ["N", "Q", "D", "M", "P"], "Invalid param name"
        if param_name not in self.params:
            return None
        else:
            return self.params[param_name]

    def get_level(self):
        return len(self.moves_from_root)

    def get_state(self):
        return current_position(self.input_board_fen, self.moves_from_root)


class LeelaSubgoal:
    def __init__(self, input_node_data, target_node_data):
        self.input_node_data = input_node_data
        self.target_node_data = target_node_data
        self.dist_from_input = self.target_node_data.get_level() - self.input_node_data.get_level()
        self.N = self.target_node_data.get_numerical_param("N")


class LeelaGMLTree:
    """Wrapper for networkx graph"""

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
            node_info = self.graph.nodes[node]
            data = LeelaNodeData(node, moves_from_root, node_info, self.input_board)
            self.graph.nodes[node]["data"] = data
            return data

    def visualize_states_graph(self) -> None:
        fig, ax = plt.subplots(figsize=(50, 50))
        pos = graphviz_layout(self.graph, prog="dot")
        nx.draw(
            self.graph,
            pos=pos,
            ax=ax,
            with_labels=True,
        )
        plt.show()

    def k_successors(self, node, k):
        k_successors_idx = set(nx.dfs_postorder_nodes(self.graph, node, depth_limit=k + 1))
        if len(k_successors_idx) > 0:
            k_successors_idx.remove(node)
        k_succesors = []
        for idx in k_successors_idx:
            k_succesors.append(LeelaSubgoal(self.get_node_data(node), self.get_node_data(idx)))
        return k_succesors

    def get_parent(self, node):
        return list(self.graph.predecessors(node))[0]

    def N_count(self):
        return [int(self.graph.nodes[node]["N"]) for node in self.graph.nodes]

    def nodes_by_N(self):
        N_to_node_dict = defaultdict(list)
        for node in self.graph.nodes:
            N_to_node_dict[int(self.graph.nodes[node]["N"])].append(node)
        N_list = [int(x) for x in N_to_node_dict.keys()]
        N_list.sort(reverse=True)
        return N_to_node_dict, N_list

    def distance_to_predecessors(self, node, predecessor) -> int:
        dist = 0
        if node != predecessor:
            parent = self.get_parent(node)
            dist += 1
            while parent != predecessor:
                parent = self.get_parent(parent)
                dist += 1
        return dist


def data_trees_generator(data_path: str, create_all_states: bool = False) -> Iterator[LeelaGMLTree]:
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

                # yield LeelaGMLTree(graph_lines, input_board, create_all_states)
                try:
                    yield LeelaGMLTree(graph_lines, input_board, create_all_states)
                except nx.exception.NetworkXError:
                    raise EndOfGMLFile
        except EndOfGMLFile:
            print("End of GML file")
