from subgoal_generator.subgoal_generator import BasicChessSubgoalGenerator


class ChessStateExpander:
    def __init__(self, policy, value, subgoal_generator):
        self.policy = policy
        self.value = value
        self.subgoal_generator = subgoal_generator

    def expand_state(self, immutable_board, n_subgoals):
        subgoals = self.subgoal_generator.generate_subgoals(immutable_board, n_subgoals)
        subgoal_values = [self.value.evaluate_immutable_board(subgoal) for subgoal in subgoals]
        for

x = BasicChessSubgoalGenerator("/home/tomasz/Research/subgoal_chess_data/local_leela_models/4gpu_generator/subgoals_k=3")
x.generate_subgoals()