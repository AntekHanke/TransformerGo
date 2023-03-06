import math
import random
import time
from collections import namedtuple
from typing import Union, Callable

import chess

from data_structures.data_structures import ImmutableBoard
from mcts.node_expansion import ChessStateExpander


def score_function(node: "TreeNode", root_player: chess.Color, exploration_constant: float) -> float:
    players_score_factor = 1 if root_player == node.get_player() else -1
    exploit_score = players_score_factor * node.get_value() * node.immutable_data.probability
    explore_score = exploration_constant * math.sqrt(
        2 * math.log(node.immutable_data.parent.num_visits) / node.num_visits
    )
    return exploit_score + exploration_constant * explore_score


def expand_function(node: "TreeNode", chess_state_expander: ChessStateExpander = None, **expander_kwargs):
    assert chess_state_expander is not None, "ChessStateExpander hasn't been provided"
    subgoals = chess_state_expander.expand_state(node.immutable_data.state, **expander_kwargs)
    for subgoal in subgoals:
        details = subgoals[subgoal]
        value = details["value"]
        probability = details["path_probabilities"]["total_path_probability"].sum()
        child = TreeNode(state=subgoal, parent=node, value=value, probability=probability)
        node.children.append(child)


def mock_expand_function(node: "TreeNode"):
    board = node.immutable_data.state.to_board()
    subgoals = (
        random.sample(list(board.legal_moves), 3) if len(list(board.legal_moves)) > 3 else list(board.legal_moves)
    )
    children = []
    for subgoal in subgoals:
        board.push(subgoal)
        immutable_board = ImmutableBoard.from_board(board)
        board.pop()
        child = TreeNode(state=immutable_board, parent=node, value=random.uniform(-1, 1), probability=1 / 3)
        children.append(child)
    node.children = children


TreeNodeData = namedtuple("TreeNode", "n_id level state parent is_terminal probability")
NodeTuple = namedtuple("NodeTuple", "n_id parent_id probability value num_visits is_terminal is_expanded")


class TreeNode:
    node_counter = 0

    def __init__(
        self,
        state: ImmutableBoard,
        parent: "TreeNode",
        value: float = 0.0,
        probability: float = 1.0,
    ):
        self.immutable_data = TreeNodeData(
            n_id=TreeNode.node_counter,
            level=0 if parent is None else parent.immutable_data.level + 1,
            state=state,
            parent=parent,
            is_terminal=state.to_board().is_checkmate(),
            probability=probability,
        )
        TreeNode.node_counter += 1
        self.is_expanded = self.immutable_data.is_terminal
        self.num_visits = 1
        self.all_values = [value]
        self.children = []

    def get_player(self) -> chess.Color:
        return self.immutable_data.state.to_board().turn

    def get_value(self) -> float:
        return sum(self.all_values) / self.num_visits


class Tree:
    def __init__(
        self,
        initial_state: ImmutableBoard,
        time_limit: float = None,
        max_mcts_passes: int = None,
        exploration_constant: float = 1 / math.sqrt(2),
        score_function: Callable[[TreeNode, chess.Color, float], float] = score_function,
        expand_function: Callable[[TreeNode, ...], None] = expand_function,
    ):
        assert initial_state is not None, "Initial state is None"
        self.root = TreeNode(state=initial_state, parent=None)
        self.root_player = self.root.get_player()
        self.node_list = [self.root]
        self.exploration_constant = exploration_constant
        self.score = score_function
        self.expand = expand_function

        assert (
            time_limit is not None or max_mcts_passes is not None
        ), "Can't have both time_limit and max_mcts_passes set to None"
        self.time_limit = time_limit
        self.max_mcts_passes = max_mcts_passes

    def mcts(self) -> dict:
        if self.time_limit is not None:
            time_limit = time.time() + self.time_limit
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
        return {"best_child": best_child.immutable_data.state, "expected_value": best_child.get_value()}

    def execute_mcts_pass(self):
        node = self.tree_traversal(self.root)
        value = node.get_value()
        self.backpropogate(node, value)

    def tree_traversal(self, node: TreeNode) -> TreeNode:
        while not node.immutable_data.is_terminal:
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
            node.all_values.append(value)
            node = node.immutable_data.parent

    def get_best_child(self, node: TreeNode, exploration_constant: float) -> TreeNode:
        best_score = float("-inf")
        best_nodes = []
        for child in node.children:
            node_score = self.score(node=child, root_player=self.root_player, exploration_constant=exploration_constant)
            if node_score > best_score:
                best_score = node_score
                best_nodes = [child]
            elif node_score == best_score:
                best_nodes.append(child)
        return random.choice(best_nodes)

    def to_list(self):
        tree_list = []
        for node in self.node_list:
            if node.immutable_data.parent is None:
                parent_id = None
            else:
                parent_id = node.immutable_data.parent.immutable_data.n_id
            node_tuple = NodeTuple(
                n_id=node.immutable_data.n_id,
                parent_id=parent_id,
                probability=node.immutable_data.probability,
                value=node.get_value(),
                num_visits=node.num_visits,
                is_terminal=node.immutable_data.is_terminal,
                is_expanded=node.is_expanded,
            )
            tree_list.append(node_tuple)
        return tree_list
