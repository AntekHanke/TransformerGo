import os
from pathlib import Path

from matplotlib.pyplot import savefig
from pyvis.network import Network

from configures.global_config import TREE_DISPLAY_LEVEL_DISTANCE_FACTOR, TREE_DISPLAY_SCORE_FACTOR
from mcts.mcts import Tree, score_function
from utils.data_utils import immutable_boards_to_img


def mcts_tree_network(tree: Tree, target_path: str, target_name: str, with_images: bool = False):
    Path(target_path).mkdir(parents=True, exist_ok=True)
    os.chdir(target_path)
    if with_images:
        Path("images").mkdir(parents=True, exist_ok=True)

    network = Network(layout=True, directed=True, height="70vw")

    for node in tree.node_list:
        n_id = node.immutable_data.n_id

        if node.immutable_data.parent is not None:
            path = " ".join([x.uci() for x in node.immutable_data.parent.paths_to_children[node.immutable_data.state]])
            score = TREE_DISPLAY_SCORE_FACTOR * score_function(node, tree.root_player, 0)
            label = f"s (x{TREE_DISPLAY_SCORE_FACTOR}): {score:.3f}\nv: {node.get_value():.3f}\np: {node.immutable_data.probability:.3f}\n{path}"
        else:
            path = " ".join([x.uci() for x in tree.root.root_final_path])
            label = f"best path: {path}\nv = {node.get_value():.3f}\np: {node.immutable_data.probability:.3f}"
        level = node.immutable_data.level
        if with_images:
            immutable_boards_to_img(immutable_boards=[node.immutable_data.state], descriptions=[""])
            image_path = os.path.join("images", f"image_{n_id}.png")
            savefig(image_path)
            network.add_node(
                n_id=n_id,
                label=label,
                level=level * TREE_DISPLAY_LEVEL_DISTANCE_FACTOR,
                shape="image",
                image=image_path,
                size=50,
            )
        else:
            network.add_node(n_id=n_id, label=label, level=level)
        if node.immutable_data.parent is not None:
            network.add_edge(
                source=node.immutable_data.parent.immutable_data.n_id,
                to=n_id,
                title=f"{node.immutable_data.probability:.4f}",
            )
    network.save_graph(f"{target_name}.html")
