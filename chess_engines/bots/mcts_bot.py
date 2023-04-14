import math
import os
import pickle
from datetime import datetime, date
from pathlib import Path
from typing import Optional, List

import chess

from chess_engines.bots.basic_chess_engines import ChessEngine
from data_structures.data_structures import ImmutableBoard
from mcts.mcts import Tree, StandardExpandFunction, score_function, LeelaExpandFunction
from mcts.mcts_tree_network import mcts_tree_network
from mcts.node_expansion import ChessStateExpander
from metric_logging import log_object
from policy.chess_policy import LCZeroPolicy
from policy.cllp import CLLP
from subgoal_generator.subgoal_generator import BasicChessSubgoalGenerator
from value.chess_value import LCZeroValue


class MCTSChessEngine(ChessEngine):
    def propose_best_moves(
        self, current_state: chess.Board, number_of_moves: int = 0, history: List[chess.Move] = None
    ) -> Optional[str]:
        self.moves_count += 1
        immutable_state = ImmutableBoard.from_board(current_state)

        if self.moves_count <= 2:
            return LCZeroPolicy().sample_move(immutable_board=immutable_state).uci()
        else:
            tree = Tree(
                initial_state=immutable_state,
                time_limit=self.time_limit,
                max_mcts_passes=self.max_mcts_passes,
                exploration_constant=self.exploration_constant,
                score_function=score_function,
                expand_function_or_class=self.expand_function,
                output_root_values_list=True,
                log_root_data=False,
            )
            mcts_output = tree.mcts()
            self.tree_count += 1

            if self.log_trees:
                mcts_tree_network(
                    tree=tree,
                    target_path=os.path.join(self.out_dir, f"tree_{self.tree_count}"),
                    target_name="tree",
                    with_images=True,
                )
                with open(os.path.join(self.out_dir, f"tree_{self.tree_count}.pkl"), "wb") as f:
                    pickle.dump(tree.to_list(), f)

            return mcts_output["best_path"][0].uci()

    def new_game(self) -> None:
        self.moves_count = 0
        self.tree_count = 0
        if self.log_trees:
            today = date.today()
            now = datetime.now()
            self.out_dir = os.path.join(
                self.log_dir, str(today), f"{now.hour}_{now.minute}_{now.second}", f"{self.name}"
            )
            Path(self.out_dir).mkdir(parents=True, exist_ok=True)


class SubgoalMCTSChessEngine(MCTSChessEngine):
    def __init__(
        self,
        time_limit: float = 120,
        max_mcts_passes: int = 10,
        exploration_constant: float = 1 / math.sqrt(2),
        generator_path: str = None,
        cllp_path: str = None,
        cllp_num_beams: int = None,
        cllp_num_return_sequences: int = None,
        generator_num_beams: int = None,
        subgoal_distance_k: int = 3,
        generator_num_subgoals: int = None,
        generator_num_subgoals_first_layer: int = None,
        sort_subgoals_by: str = None,
        subgoal_probs_opponent_only: bool = None,
        num_top_subgoals: int = None,
        num_top_subgoals_first_layer: int = None,
        log_trees: bool = False,
        log_dir: str = None,
    ):
        self.name = "Subgoal_MCTS"
        self.time_limit = time_limit
        self.max_mcts_passes = max_mcts_passes
        self.exploration_constant = exploration_constant
        self.generator_path = generator_path
        self.cllp_path = cllp_path
        self.cllp_num_beams = cllp_num_beams
        self.cllp_num_return_sequences = cllp_num_return_sequences
        self.generator_num_beams = generator_num_beams
        self.generator_num_subgoals = generator_num_subgoals
        self.generator_num_subgoals_first_layer = generator_num_subgoals_first_layer
        self.subgoal_distance_k = subgoal_distance_k
        self.sort_subgoals_by = sort_subgoals_by
        self.subgoal_probs_opponent_only = subgoal_probs_opponent_only
        self.num_top_subgoals = num_top_subgoals
        self.num_top_subgoals_first_layer = num_top_subgoals_first_layer
        self.log_trees = log_trees
        self.log_dir = log_dir

        self.log_all_params()

        generator = BasicChessSubgoalGenerator(self.generator_path)
        cllp = CLLP(self.cllp_path)
        self.chess_states_expander = ChessStateExpander(LCZeroPolicy, LCZeroValue, generator, cllp)
        self.expand_function = StandardExpandFunction(
            chess_state_expander_or_class=self.chess_states_expander,
            cllp_num_beams=self.cllp_num_beams,
            cllp_num_return_sequences=self.cllp_num_return_sequences,
            generator_num_beams=self.generator_num_beams,
            generator_num_subgoals=self.generator_num_subgoals,
            generator_num_subgoals_first_layer=self.generator_num_subgoals_first_layer,
            subgoal_distance_k=self.subgoal_distance_k,
            sort_subgoals_by=self.sort_subgoals_by,
            subgoal_probs_opponent_only=self.subgoal_probs_opponent_only,
            num_top_subgoals=self.num_top_subgoals,
            num_top_subgoals_first_layer=self.num_top_subgoals_first_layer,
        )

        log_object("Status", "ready")


class VanillaMCTSChessEngine(MCTSChessEngine):
    def __init__(
        self,
        time_limit: float = 120,
        max_mcts_passes: int = 10,
        exploration_constant: float = 1 / math.sqrt(2),
        policy_num_moves: int = None,
        policy_num_beams: int = None,
        log_trees: bool = False,
        log_dir: str = None,
    ):
        self.name = "Vanilla_MCTS"
        self.time_limit = time_limit
        self.max_mcts_passes = max_mcts_passes
        self.exploration_constant = exploration_constant
        self.policy_num_moves = policy_num_moves
        self.policy_num_beams = policy_num_beams
        self.log_trees = log_trees
        self.log_dir = log_dir

        self.log_all_params()
        self.game_count = 0

        self.expand_function = LeelaExpandFunction(
            num_return_moves=self.policy_num_moves, num_beams=self.policy_num_beams
        )
