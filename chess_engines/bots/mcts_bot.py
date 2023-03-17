import math
from typing import Optional, List

import chess

from chess_engines.bots.basic_chess_engines import ChessEngine
from data_structures.data_structures import ImmutableBoard
from mcts.mcts import Tree, StandardExpandFunction, score_function
from mcts.node_expansion import ChessStateExpander
from metric_logging import log_param
from policy.chess_policy import LCZeroPolicy
from policy.cllp import CLLP
from subgoal_generator.subgoal_generator import BasicChessSubgoalGenerator
from value.chess_value import LCZeroValue


class MCTSChessEngine(ChessEngine):
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
        generator_num_subgoals: int = None,
        sort_subgoals_by: str = None,
        num_top_subgoals: int = None,
    ):
        self.name = "MCTS"
        self.time_limit = time_limit
        self.max_mcts_passes = max_mcts_passes
        self.exploration_constant = exploration_constant
        self.generator_path = generator_path
        self.cllp_path = cllp_path
        self.cllp_num_beams = cllp_num_beams
        self.cllp_num_return_sequences = cllp_num_return_sequences
        self.generator_num_beams = generator_num_beams
        self.generator_num_subgoals = generator_num_subgoals
        self.sort_subgoals_by = sort_subgoals_by
        self.num_top_subgoals = num_top_subgoals

        self.networks = None

        for attr, value in self.__dict__.items():
            log_param(f"MCTS_{attr}", str(value))


    def propose_best_moves(
        self, current_state: chess.Board, number_of_moves: int = 0, history: List[chess.Move] = None
    ) -> Optional[str]:
        immutable_state = ImmutableBoard.from_board(current_state)

        if self.networks is None:
            generator = BasicChessSubgoalGenerator(self.generator_path)
            cllp = CLLP(self.cllp_path)
            self.chess_states_expander = ChessStateExpander(LCZeroPolicy, LCZeroValue, generator, cllp)
            self.expand_function = StandardExpandFunction(
                self.chess_states_expander,
                self.cllp_num_beams,
                self.cllp_num_return_sequences,
                self.generator_num_beams,
                self.generator_num_subgoals,
                self.sort_subgoals_by,
                self.num_top_subgoals,
            )
            self.networks = True

        tree = Tree(
            initial_state=immutable_state,
            time_limit=self.time_limit,
            max_mcts_passes=self.max_mcts_passes,
            exploration_constant=self.exploration_constant,
            score_function=score_function,
            expand_function_or_class=self.expand_function,
        )
        mcts_output = tree.mcts()
        return mcts_output["best_path"][0].uci()

    def new_game(self) -> None:
        pass
