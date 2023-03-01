from pyvis.network import Network

from mcts.mcts import Tree


def mcts_tree_network(tree: Tree, target_file: str):
    network = Network(layout=True, directed=True)
    for node in tree.node_list:
        value = node.total_value / node.num_visits
        network.add_node(n_id=node.id, label=f"{value:.4f}", level=node.level)
        if node.parent is not None:
            network.add_edge(source=node.parent.id, to=node.id, title=f"{node.probability:.4f}")
    network.save_graph(target_file)
