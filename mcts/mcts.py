import math
import random
import time
from collections import namedtuple
from typing import Callable, Type, List, Union

import chess

from chess_engines.third_party.stockfish import StockfishEngine
from data_structures.data_structures import ImmutableBoard
from mcts.node_expansion import ChessStateExpander
from metric_logging import (
    log_value_without_step,
    accumulator_to_logger,
    log_value_to_accumulate,
    log_value,
    log_value_to_average,
    log_param,
    log_object,
)
from policy.chess_policy import LCZeroPolicy
from utils.data_utils import immutable_boards_to_img
from value.chess_value import LCZeroValue


def score_function(node: "TreeNode", root_player: chess.Color, exploration_constant: float) -> float:
    players_score_factor = 1 if root_player == node.get_player() else -1
    exploit_score = players_score_factor * node.get_value() * node.immutable_data.probability
    explore_score = math.sqrt(2 * math.log(node.immutable_data.parent.num_visits) / node.num_visits)
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
        generator_num_subgoals_first_layer: int = None,
        subgoal_distance_k: int = 3,
        sort_subgoals_by: str = None,
        num_top_subgoals: int = None,
        num_top_subgoals_first_layer: int = None,
        debug_mode: bool = True,
    ):
        if isinstance(chess_state_expander_or_class, ChessStateExpander):
            self.chess_state_expander = chess_state_expander_or_class
        else:
            self.chess_state_expander = chess_state_expander_or_class()
        self.cllp_num_beams = cllp_num_beams
        self.cllp_num_return_sequences = cllp_num_return_sequences
        self.generator_num_beams = generator_num_beams
        self.generator_num_subgoals = generator_num_subgoals
        self.generator_num_subgoals_first_layer = (
            generator_num_subgoals_first_layer
            if generator_num_subgoals_first_layer is not None
            else generator_num_subgoals_first_layer
        )
        self.subgoal_distance_k = subgoal_distance_k
        self.sort_subgoals_by = sort_subgoals_by
        self.num_top_subgoals = num_top_subgoals
        self.num_top_subgoals_first_layer = (
            num_top_subgoals_first_layer if num_top_subgoals_first_layer is not None else num_top_subgoals
        )
        self.debug_mode = debug_mode
        self.leela = LCZeroPolicy()

        log_param("Parameters for ", self.__class__.__name__)
        log_param("cllp_num_beams", self.cllp_num_beams)
        log_param("cllp_num_return_sequences", self.cllp_num_return_sequences)
        log_param("generator_num_beams", self.generator_num_beams)
        log_param("generator_num_subgoals", self.generator_num_subgoals)
        log_param("subgoal_distance_k", self.subgoal_distance_k)
        log_param("sort_subgoals_by", self.sort_subgoals_by)
        log_param("num_top_subgoals", self.num_top_subgoals)

    def expand_function(self, node: "TreeNode", **kwargs) -> None:
        assert self.chess_state_expander is not None, "ChessStateExpander hasn't been provided"
        first_layer = node.immutable_data.level == 0
        generator_num_subgoals = self.generator_num_subgoals_first_layer if first_layer else self.generator_num_subgoals
        subgoals, subgoals_info = self.chess_state_expander.expand_state(
            input_immutable_board=node.immutable_data.state,
            siblings_states=node.get_siblings_states(),
            cllp_num_beams=self.cllp_num_beams,
            cllp_num_return_sequences=self.cllp_num_return_sequences,
            generator_num_beams=self.generator_num_beams,
            generator_num_subgoals=generator_num_subgoals,
            subgoal_distance_k=self.subgoal_distance_k,
            sort_subgoals_by=self.sort_subgoals_by,
        )
        num_top_subgoals = self.num_top_subgoals_first_layer if first_layer else self.num_top_subgoals
        subgoals = subgoals[:num_top_subgoals]
        if not subgoals:
            log_object("Subgoal generation failed", node.immutable_data.state.fen())
            log_value_to_accumulate("Number of node expansions failed", 1)
            log_value_to_average("Fraction of node expansions failed", 1)
        else:
            log_value_to_average("Fraction of node expansions failed", 0)
        for subgoal in subgoals:
            details = subgoals_info[subgoal]
            value = details["value"]
            probability = sum(
                [path_statistics["total_path_probability"] for path_statistics in details["path_probabilities"]]
            )
            child = TreeNode(state=subgoal, parent=node, value=value, probability=probability)
            node.children.append(child)
            node.paths_to_children[subgoal] = details["path_with_highest_min_probability"]

        if self.debug_mode:
            node.best_moves_by_leela = self.leela.get_best_moves(
                node.immutable_data.state, num_return_sequences=self.num_top_subgoals
            )
            moves_by_subgoals = [node.paths_to_children[subgoal][0] for subgoal in subgoals]
            top_n_moves = []
            for idx, move in enumerate(node.best_moves_by_leela):
                top_n_moves.append(move)
                intersection = set(top_n_moves).intersection(set(moves_by_subgoals))
                log_value_to_average(f"moves_len_intersect_subgoal_top_{idx+1}_leela", len(intersection))
                log_value_to_average(f"moves_intersect_subgoal_top_{idx+1}_leela", len(intersection) > 0)


class LeelaExpandFunction(ExpandFunction):
    def __init__(
        self,
        num_return_moves: int,
        num_beams: int,
        num_return_moves_first_layer: int = None,
    ):
        self.policy = LCZeroPolicy()
        self.value = LCZeroValue()
        self.num_return_moves = num_return_moves
        self.num_return_moves_first_layer = (
            num_return_moves_first_layer if num_return_moves_first_layer is not None else num_return_moves
        )
        self.num_beams = num_beams

    def expand_function(self, node: "TreeNode", **kwargs) -> None:
        num_return_sequences = (
            self.num_return_moves_first_layer if node.immutable_data.level == 0 else self.num_return_moves
        )
        moves, probs = self.policy.get_best_moves(
            immutable_board=node.immutable_data.state,
            num_return_sequences=num_return_sequences,
            num_beams=self.num_beams,
            return_probs=True,
        )
        for move, prob in zip(moves, probs):
            new_board = node.immutable_data.state.to_board()
            new_board.push(move)
            new_immutable_board = ImmutableBoard.from_board(new_board)
            value = self.value.evaluate_immutable_board(new_immutable_board)
            child = TreeNode(state=new_immutable_board, parent=node, value=value, probability=prob)
            node.children.append(child)
            node.paths_to_children[new_immutable_board] = [move]


TreeNodeData = namedtuple("TreeNode", "n_id level state parent is_terminal probability")
NodeTuple = namedtuple(
    "NodeTuple", "n_id parent_id probability value num_visits is_terminal is_expanded not_expandable state"
)


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
        self.not_expandable = self.immutable_data.is_terminal
        self.is_expanded = self.immutable_data.is_terminal
        self.num_visits = 1
        self.all_values = [value]
        self.children = []
        self.paths_to_children = {}
        self.root_final_path = None
        self.best_moves_by_leela = None

    def get_player(self) -> chess.Color:
        return self.immutable_data.state.to_board().turn

    def get_value(self) -> float:
        return sum(self.all_values) / self.num_visits

    def get_siblings_states(self) -> List[ImmutableBoard]:
        if self.immutable_data.parent is None:
            return [self.immutable_data.state]
        return [node.immutable_data.state for node in self.immutable_data.parent.children]

    def to_named_tuple(self) -> NodeTuple:
        parent_id = self.immutable_data.parent.immutable_data.n_id if self.immutable_data.parent is not None else None
        return NodeTuple(
            n_id=self.immutable_data.n_id,
            parent_id=parent_id,
            probability=self.immutable_data.probability,
            value=self.get_value(),
            num_visits=self.num_visits,
            is_terminal=self.immutable_data.is_terminal,
            is_expanded=self.is_expanded,
            not_expandable=self.not_expandable,
            state=self.immutable_data.state,
        )


class Tree:
    total_mcts_passes_counter = 0
    trees_counter = 0

    def __init__(
        self,
        initial_state: ImmutableBoard,
        time_limit: float = None,
        max_mcts_passes: int = None,
        exploration_constant: float = 1 / math.sqrt(2),
        score_function: Callable[[TreeNode, chess.Color, float], float] = score_function,
        expand_function_or_class: Union[Type[ExpandFunction], ExpandFunction] = None,
        output_root_values_list: bool = True,
        log_root_data: bool = True,
    ):
        assert initial_state is not None, "Initial state is None"
        self.root = TreeNode(state=initial_state, parent=None)
        self.root_player = self.root.get_player()
        self.node_list = [self.root]
        self.exploration_constant = exploration_constant
        self.score_function = score_function
        if isinstance(expand_function_or_class, ExpandFunction):
            self.expand_function = expand_function_or_class
        else:
            self.expand_function = expand_function_or_class()
        self.mcts_passes_counter = 0
        self.output_root_values_list = output_root_values_list
        if output_root_values_list:
            self.root_values_list = []
        self.log_root_data = log_root_data

        assert (
            time_limit is not None or max_mcts_passes is not None
        ), "Can't have both time_limit and max_mcts_passes set to None"
        self.time_limit = time_limit
        self.max_mcts_passes = max_mcts_passes

        self.lc0_policy = LCZeroPolicy()
        self.stockfish = StockfishEngine(depth_limit=20)
        Tree.trees_counter += 1

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
        root_moves = [x[0] for x in self.root.paths_to_children.values()]
        root_subgoals = [x.immutable_data.state for x in self.root.children]
        output_dir = {
            "best_node": best_child,
            "best_path": best_path,
            "best_child": best_child.immutable_data.state,
            "expected_value": best_child.get_value(),
            "root_moves": root_moves,
            "root_subgoals": root_subgoals,
            "leela_best_move": self.lc0_policy.get_best_moves(self.root.immutable_data.state, None, 1)[0],
        }
        self.root.root_final_path = best_path
        if self.output_root_values_list:
            output_dir["root_values_list"] = self.root_values_list
        if self.log_root_data:
            final_move = best_path[0]
            log_object("root_moves", f"root_moves_{Tree.trees_counter} {[str(x) for x in root_moves]}")
            subgoal_values = self.stockfish.evaluate_boards_in_parallel(root_subgoals)
            input_board_value = self.stockfish.evaluate_immutable_board(self.root.immutable_data.state)
            subgoal_mcts_values = [round(x.get_value(), 2) for x in self.root.children]
            move_boards = []
            for move in root_moves:
                board = self.root.immutable_data.state.to_board()
                board.push(move)
                move_boards.append(ImmutableBoard.from_board(board))
            move_values = self.stockfish.evaluate_boards_in_parallel(move_boards)
            descriptions = [
                f"m: {a} v_s: {b} v_m: {c} t_v: {d}"
                for a, b, c, d in zip(root_moves, subgoal_values, move_values, subgoal_mcts_values)
            ]
            fig = immutable_boards_to_img(
                [self.root.immutable_data.state] + root_subgoals,
                [f"input v: {input_board_value} t_v: {round(self.root.get_value(),2)} chosen m: {final_move}"]
                + descriptions,
            )
            log_object("root_subgoals", fig)
        return output_dir

    def execute_mcts_pass(self):
        self.mcts_passes_counter += 1
        Tree.total_mcts_passes_counter += 1
        nodes_before_pass = len(self.node_list)
        log_value_without_step("MCTS passes", self.mcts_passes_counter)
        node = self.tree_traversal(self.root)
        value = node.get_value()
        self.backpropogate(node, value)
        log_value(
            "Nodes in a single pass",
            Tree.total_mcts_passes_counter,
            len(self.node_list) - nodes_before_pass,
        )
        accumulator_to_logger(Tree.total_mcts_passes_counter)
        if self.output_root_values_list:
            self.root_values_list.append(self.root.get_value())

    def tree_traversal(self, node: TreeNode) -> TreeNode:
        while not node.not_expandable:
            if node.is_expanded:
                node = self.get_best_child(node, self.exploration_constant)
            else:
                self.expand_function.expand_function(node=node)
                self.node_list += node.children
                node.is_expanded = True
                if not node.children:
                    node.not_expandable = True
                    return node
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
        assert node.children, "Node provided to get_best_child method has no children"
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
            node_tuple = node.to_named_tuple()
            tree_list.append(node_tuple)
        return tree_list
