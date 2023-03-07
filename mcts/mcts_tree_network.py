import os

from pyvis.network import Network

from mcts.mcts import Tree


def mcts_tree_network(tree: Tree, target_path: str, target_file: str):
    network = Network(layout=True, directed=True)
    for node in tree.node_list:
        value = node.get_value()
        network.add_node(n_id=node.immutable_data.n_id, label=f"{value:.4f}", level=node.immutable_data.level)
        if node.immutable_data.parent is not None:
            network.add_edge(
                source=node.immutable_data.parent.immutable_data.n_id,
                to=node.immutable_data.n_id,
                title=f"{node.immutable_data.probability:.4f}",
            )
    os.chdir(target_path)
    network.save_graph(target_file)
