import os
from pathlib import Path

from matplotlib.pyplot import savefig
from pyvis.network import Network

from mcts.mcts import Tree
from utils.data_utils import immutable_boards_to_img


def mcts_tree_network(tree: Tree, target_path: str, target_name: str, with_images: bool = False):
    Path(target_path).mkdir(parents=True, exist_ok=True)
    os.chdir(target_path)
    if with_images:
        Path("images").mkdir(parents=True, exist_ok=True)

    network = Network(layout=True, directed=True)

    for node in tree.node_list:
        n_id = node.immutable_data.n_id
        label = f"{node.get_value():.4f}"
        level = node.immutable_data.level
        if with_images:
            immutable_boards_to_img(immutable_boards=[node.immutable_data.state], descriptions=[""])
            image_path = os.path.join("images", f"image_{n_id}.png")
            savefig(image_path)
            network.add_node(n_id=n_id, label=label, level=level, shape="image", image=image_path, size=50)
        else:
            network.add_node(n_id=n_id, label=label, level=level)
        if node.immutable_data.parent is not None:
            network.add_edge(
                source=node.immutable_data.parent.immutable_data.n_id,
                to=n_id,
                title=f"{node.immutable_data.probability:.4f}",
            )
    network.save_graph(f"{target_name}.html")
