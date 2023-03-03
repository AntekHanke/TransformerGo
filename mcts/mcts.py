import math
import random
import time
from collections import namedtuple
from typing import Union

import chess

from data_structures.data_structures import ImmutableBoard
from mcts.node_expansion import ChessStateExpander


def score_function(node: "TreeNode", root_player: chess.Color, exploration_constant: float) -> float:
    players_score_factor = 1 if root_player == node.get_player() else -1
    exploit_score = players_score_factor * node.total_value / node.num_visits * node.probability
    explore_score = exploration_constant * math.sqrt(2 * math.log(node.parent.num_visits) / node.num_visits)
    return exploit_score + exploration_constant * explore_score


def expand_function(node: "TreeNode", chess_state_expander: ChessStateExpander = None, **expander_kwargs):
    assert chess_state_expander is not None, "ChessStateExpander hasn't been provided"
    subgoals = chess_state_expander.expand_state(node.state, **expander_kwargs)
    for subgoal in subgoals:
        details = subgoals[subgoal]
        value = details["value"]
        probability = details["path_probabilities"]["total_path_probability"].sum()
        child = TreeNode(state=subgoal, parent=node, value=value, probability=probability)
        node.children.append(child)


def mock_expand_function(node: "TreeNode"):
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
        child = TreeNode(state=immutable_board, parent=node, value=random.uniform(-1, 1), probability=1 / 3)
        children.append(child)
    node.children = children


TreeNodeData = namedtuple("TreeNode", "n_id level state parent is_terminal probability")
NodeTuple = namedtuple("NodeTuple", "n_id parent_id probability total_value num_visits is_terminal is_expanded")


class TreeNode(TreeNodeData):
    node_counter = 0

    def __new__(
        cls,
        state: ImmutableBoard,
        parent: "TreeNode",
        value=0.0,
        probability=1.0,
    ):
        self = super(TreeNode, cls).__new__(
            cls,
            n_id=TreeNode.node_counter,
            level=0 if parent is None else parent.level + 1,
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
        initial_state: ImmutableBoard,
        time_limit: float = None,
        max_mcts_passes: int = None,
        exploration_constant: float = 1 / math.sqrt(2),
        score_function=score_function,
        expand_function=expand_function,
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
        return {"best_child": best_child.state, "expected_value": best_child.total_value / best_child.num_visits}

    def execute_mcts_pass(self):
        node = self.tree_traversal(self.root)
        value = node.total_value
        self.backpropogate(node, value)

    def tree_traversal(self, node: TreeNode) -> TreeNode:
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
            if node.parent is None:
                parent_id = None
            else:
                parent_id = node.parent.n_id
            node_tuple = NodeTuple(
                n_id=node.n_id,
                parent_id=parent_id,
                probability=node.probability,
                total_value=node.total_value,
                num_visits=node.num_visits,
                is_terminal=node.is_terminal,
                is_expanded=node.is_expanded,
            )
            tree_list.append(node_tuple)
        return tree_list
