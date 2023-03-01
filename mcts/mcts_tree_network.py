from pyvis.network import Network
from mcts.mcts import Tree


def mcts_tree_network(tree: Tree, target_file: str):
    network = Network(layout=True, directed=True)
    for node in tree.node_list:
        network.add_node(n_id=node.id, label=str(node.state.to_board()), level=node.level)
        if node.parent is not None:
            network.add_edge(source=node.parent.id, to=node.id, weight=node.probability)
    network.save_graph(target_file)