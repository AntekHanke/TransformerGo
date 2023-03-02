import math
import random
import time
from collections import namedtuple
from typing import Union

import chess

from data_structures.data_structures import ImmutableBoard


def score_function(node: "TreeNode", player: chess.Color, exploration_constant: float) -> float:
    if player == node.get_player():
        players_move = 1
    else:
        players_move = -1
    return players_move * node.total_value / node.num_visits * node.probability + exploration_constant * math.sqrt(
        2 * math.log(node.parent.num_visits) / node.num_visits
    )


def expansion_function(node: "TreeNode", chess_state_expander=None, **expander_kwargs):
    assert chess_state_expander is not None, "ChessStateExpander hasn't been provided"
    subgoals = chess_state_expander.expand_state(node.state, **expander_kwargs)
    for subgoal in subgoals:
        details = subgoals[subgoal]
        value = details["value"]
        probability = details["path_probabilities"]["total_path_probability"].sum()
        child = TreeNode(subgoal, node, value, probability)
        node.children.append(child)


def mock_expansion_function(node: "TreeNode"):
    board = node.state.to_board()
    if len(list(board.legal_moves)) > 3:
        subgoals = random.sample(list(board.legal_moves), 3)
    else:
        subgoals = list(board.legal_moves)
    children = []
    for subgoal in subgoals:
        board.push(subgoal)
        immutable_board = ImmutableBoard.from_board(board)
        board.pop()
        child = TreeNode(immutable_board, node, random.uniform(-1, 1), 1 / 3)
        children.append(child)
    node.children = children


TreeNodeData = namedtuple("TreeNode", "id level state parent is_terminal probability")


class TreeNode(TreeNodeData):
    node_counter = 0

    def __new__(cls, state: ImmutableBoard, parent: "TreeNode", value=0.0, probability=1.0):
        if parent is None:
            level = 0
        else:
            level = parent.level + 1
        self = super(TreeNode, cls).__new__(
            cls,
            id=TreeNode.node_counter,
            level=level,
            state=state,
            parent=parent,
            is_terminal=state.to_board().is_checkmate(),
            probability=probability,
        )
        TreeNode.node_counter += 1
        self.is_expanded = self.is_terminal
        self.num_visits = 1
        self.total_value = value
        self.children = []
        return self

    def get_player(self) -> chess.Color:
        return self.state.to_board().turn


class Tree:
    def __init__(
        self,
        initial_state,
        time_limit: int = None,
        max_mcts_passes: int = None,
        exploration_constant: float = 1 / math.sqrt(2),
        score=score_function,
        expand=expansion_function,
    ):
        assert initial_state is not None, "Initial state is None"
        self.root = TreeNode(initial_state, None)
        self.player = self.root.get_player()
        self.node_list = [self.root]
        self.exploration_constant = exploration_constant
        self.score = score
        self.expand = expand

        assert (
            time_limit is not None or max_mcts_passes is not None
        ), "Can't have both time_limit and max_mcts_passes set to None"
        self.time_limit = time_limit
        self.max_mcts_passes = max_mcts_passes

    def mcts(self, need_details=False) -> Union[ImmutableBoard, dict]:
        if self.time_limit is not None:
            time_limit = time.time() + self.time_limit / 1000
            if self.max_mcts_passes is not None:
                for i in range(self.max_mcts_passes):
                    self.execute_mcts_pass()
                    if time.time() >= time_limit:
                        break
            else:
                while time.time() < time_limit:
                    self.execute_mcts_pass()
        else:
            if self.max_mcts_passes is not None:
                for i in range(self.max_mcts_passes):
                    self.execute_mcts_pass()

        best_child = self.get_best_child(self.root, 0)
        if need_details:
            return {"subgoal": best_child.state, "expected_value": best_child.total_value / best_child.num_visits}
        else:
            return best_child.state

    def execute_mcts_pass(self):
        node = self.select_node(self.root)
        value = node.total_value
        self.backpropogate(node, value)

    def select_node(self, node: TreeNode) -> TreeNode:
        while not node.is_terminal:
            if node.is_expanded:
                node = self.get_best_child(node, self.exploration_constant)
            else:
                self.expand(node)
                self.node_list += node.children
                node.is_expanded = True
                return self.get_best_child(node, self.exploration_constant)
        return node

    def backpropogate(self, node: TreeNode, value: float):
        while node is not None:
            node.num_visits += 1
            node.total_value += value
            node = node.parent

    def get_best_child(self, node: TreeNode, exploration_constant: float) -> TreeNode:
        best_score = float("-inf")
        best_nodes = []
        for child in node.children:
            node_score = self.score(child, self.player, exploration_constant)
            if node_score > best_score:
                best_score = node_score
                best_nodes = [child]
            elif node_score == best_score:
                best_nodes.append(child)
        return random.choice(best_nodes)
