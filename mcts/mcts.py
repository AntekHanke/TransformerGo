import math
import random
import time
from collections import namedtuple
from typing import Callable, Type, List, Union

import chess

from data_structures.data_structures import ImmutableBoard
from mcts.node_expansion import ChessStateExpander
from metric_logging import (
    log_value_without_step,
    accumulator_to_logger,
    log_value_to_accumulate,
    log_value,
    log_value_to_average,
    log_param,
)
from policy.chess_policy import ChessPolicy
from value.chess_value import ChessValue


def score_function(node: "TreeNode", root_player: chess.Color, exploration_constant: float) -> float:
    players_score_factor = 1 if root_player == node.get_player() else -1
    exploit_score = players_score_factor * node.get_value() * node.immutable_data.probability
    explore_score = exploration_constant * math.sqrt(
        2 * math.log(node.immutable_data.parent.num_visits) / node.num_visits
    )
    return exploit_score + exploration_constant * explore_score


class ExpandFunction:
    def expand_function(self, node: "TreeNode", **kwargs):
        raise NotImplementedError


class StandardExpandFunction(ExpandFunction):
    def __init__(
        self,
        chess_state_expander_or_class: Union[Type[ChessStateExpander], ChessStateExpander] = None,
        cllp_num_beams: int = None,
        cllp_num_return_sequences: int = None,
        generator_num_beams: int = None,
        generator_num_subgoals: int = None,
        subgoal_distance_k: int = 3,
        sort_subgoals_by: str = None,
        num_top_subgoals: int = None,
    ):
        if isinstance(chess_state_expander_or_class, ChessStateExpander):
            self.chess_state_expander = chess_state_expander_or_class
        else:
            self.chess_state_expander = chess_state_expander_or_class()
        self.cllp_num_beams = cllp_num_beams
        self.cllp_num_return_sequences = cllp_num_return_sequences
        self.generator_num_beams = generator_num_beams
        self.generator_num_subgoals = generator_num_subgoals
        self.subgoal_distance_k = subgoal_distance_k
        self.sort_subgoals_by = sort_subgoals_by
        self.num_top_subgoals = num_top_subgoals

        log_param("Parameters for ", self.__class__.__name__)
        log_param("cllp_num_beams", self.cllp_num_beams)
        log_param("cllp_num_return_sequences", self.cllp_num_return_sequences)
        log_param("generator_num_beams", self.generator_num_beams)
        log_param("generator_num_subgoals", self.generator_num_subgoals)
        log_param("subgoal_distance_k", self.subgoal_distance_k)
        log_param("sort_subgoals_by", self.sort_subgoals_by)
        log_param("num_top_subgoals", self.num_top_subgoals)

    def expand_function(self, node: "TreeNode", **kwargs):
        assert self.chess_state_expander is not None, "ChessStateExpander hasn't been provided"
        time_s = time.time()
        subgoals, subgoals_info = self.chess_state_expander.expand_state(
            input_immutable_board=node.immutable_data.state,
            siblings_states=node.get_siblings_states(),
            cllp_num_beams=self.cllp_num_beams,
            cllp_num_return_sequences=self.cllp_num_return_sequences,
            generator_num_beams=self.generator_num_beams,
            generator_num_subgoals=self.generator_num_subgoals,
            subgoal_distance_k=self.subgoal_distance_k,
            sort_subgoals_by=self.sort_subgoals_by,
        )
        subgoals = subgoals[: self.num_top_subgoals]
        for subgoal in subgoals:
            details = subgoals_info[subgoal]
            value = details["value"]
            probability = sum(
                [path_statistics["total_path_probability"] for path_statistics in details["path_probabilities"]]
            )
            child = TreeNode(state=subgoal, parent=node, value=value, probability=probability)
            node.children.append(child)
            node.paths_to_children[subgoal] = details["path_with_highest_min_probability"]


class PolicyOnlyExpandFunction(ExpandFunction):
    def __init__(
        self,
        chess_policy_class: Type[ChessPolicy],
        chess_value_class: Type[ChessValue],
        num_return_moves: int,
        num_beams: int,
    ):
        self.policy = chess_policy_class()
        self.value = chess_value_class()
        self.num_return_moves = num_return_moves
        self.num_beams = num_beams

    def expand_function(self, node: "TreeNode", **kwargs):
        time_s = time.time()
        moves, probs = self.policy.get_best_moves(
            immutable_board=node.immutable_data.state,
            num_return_sequences=self.num_return_moves,
            num_beams=self.num_beams,
            return_probs=True,
        )
        print(f"Expand function took {time.time() - time_s} seconds")
        for move, prob in zip(moves, probs):
            new_board = node.immutable_data.state.to_board()
            new_board.push(move)
            new_immutable_board = ImmutableBoard.from_board(new_board)
            value = self.value.evaluate_immutable_board(new_immutable_board)
            child = TreeNode(state=new_immutable_board, parent=node, value=value, probability=prob)
            node.children.append(child)


TreeNodeData = namedtuple("TreeNode", "n_id level state parent is_terminal probability")
NodeTuple = namedtuple("NodeTuple", "n_id parent_id probability value num_visits is_terminal is_expanded state")


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
            is_terminal=state.to_board().is_game_over(),
            probability=probability,
        )
        TreeNode.node_counter += 1
        log_value_to_accumulate("tree_nodes", 1)
        self.is_expanded = self.immutable_data.is_terminal
        self.num_visits = 1
        self.all_values = [value]
        self.children = []
        self.paths_to_children = {}

    def get_player(self) -> chess.Color:
        return self.immutable_data.state.to_board().turn

    def get_value(self) -> float:
        return sum(self.all_values) / self.num_visits

    def get_siblings_states(self) -> List[ImmutableBoard]:
        if self.immutable_data.parent is None:
            return [self.immutable_data.state]
        return [node.immutable_data.state for node in self.immutable_data.parent.children]


class Tree:
    def __init__(
        self,
        initial_state: ImmutableBoard,
        time_limit: float = None,
        max_mcts_passes: int = None,
        exploration_constant: float = 1 / math.sqrt(2),
        score_function: Callable[[TreeNode, chess.Color, float], float] = score_function,
        expand_function_or_class: Union[Type[ExpandFunction], ExpandFunction] = None,
        counter_initial_value: int = 0,
    ):
        assert initial_state is not None, "Initial state is None"
        self.root = TreeNode(state=initial_state, parent=None)
        self.root_player = self.root.get_player()
        self.node_list = [self.root]
        self.exploration_constant = exploration_constant
        self.score_function = score_function
        self.counter_initial_value = counter_initial_value
        if isinstance(expand_function_or_class, ExpandFunction):
            self.expand_function = expand_function_or_class
        else:
            self.expand_function = expand_function_or_class()
        self.mcts_passes_counter = 0

        assert (
            time_limit is not None or max_mcts_passes is not None
        ), "Can't have both time_limit and max_mcts_passes set to None"
        self.time_limit = time_limit
        self.max_mcts_passes = max_mcts_passes

    def mcts(self) -> dict:
        self.mcts_passes_counter = 0
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
        best_path = self.root.paths_to_children[best_child.immutable_data.state]
        return {
            "best_node": best_child,
            "best_path": best_path,
            "best_child": best_child.immutable_data.state,
            "expected_value": best_child.get_value(),
        }

    def execute_mcts_pass(self):
        self.mcts_passes_counter += 1
        nodes_before_pass = len(self.node_list)
        log_value_without_step("MCTS passes", self.mcts_passes_counter)
        node = self.tree_traversal(self.root)
        value = node.get_value()
        self.backpropogate(node, value)
        log_value("Nodes in a single pass", self.counter_initial_value + self.mcts_passes_counter, len(self.node_list) - nodes_before_pass)
        log_value_to_average("Nodes in a single pass", len(self.node_list) - nodes_before_pass)
        accumulator_to_logger(self.counter_initial_value + self.mcts_passes_counter)

    def tree_traversal(self, node: TreeNode) -> TreeNode:
        while not node.immutable_data.is_terminal:
            if node.is_expanded:
                node = self.get_best_child(node, self.exploration_constant)
            else:
                self.expand_function.expand_function(node=node)
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
            node_score = self.score_function(
                node=child, root_player=self.root_player, exploration_constant=exploration_constant
            )
            if node_score > best_score:
                best_score = node_score
                best_nodes = [child]
            elif node_score == best_score:
                best_nodes.append(child)
        return random.choice(best_nodes)

    def to_list(self):
        tree_list = []
        for node in self.node_list:
            parent_id = (
                node.immutable_data.parent.immutable_data.n_id if node.immutable_data.parent is not None else None
            )
            node_tuple = NodeTuple(
                n_id=node.immutable_data.n_id,
                parent_id=parent_id,
                probability=node.immutable_data.probability,
                value=node.get_value(),
                num_visits=node.num_visits,
                is_terminal=node.immutable_data.is_terminal,
                is_expanded=node.is_expanded,
                state=node.immutable_data.state,
            )
            tree_list.append(node_tuple)
        return tree_list
