import chess
from lczero.backends import Weights, Backend, GameState

global lczero_backend
lczero_backend = None

class LCZeroBackend:
    def __init__(self):
        self.weights = Weights()
        self.backend = Backend(weights=self.weights)

    def evaluate_immutable_board(self, immutable_board):
        game_state = GameState(fen=immutable_board.fen())
        input = game_state.as_input(self.backend)
        output = self.backend.evaluate(input)[0]
        return output.q()

    def get_policy_distribution(self, immutable_board):
        game_state = GameState(fen=immutable_board.fen())
        input = game_state.as_input(self.backend)
        output = self.backend.evaluate(input)[0]
        moves = list(zip(game_state.moves(), output.p_softmax(*game_state.policy_indices())))
        sorted_moves = sorted(moves, key=lambda x: x[1], reverse=True)
        return sorted_moves

    def policy_distribution_dict(self, immutable_board):
        sorted_moves = self.get_policy_distribution(immutable_board)
        return {move[0]: move[1] for move in sorted_moves}


def get_lczero_backend():
    """Generates a new LCZero backend if one doesn't exist, otherwise returns the existing one."""
    global lczero_backend
    if lczero_backend is None:
        lczero_backend = LCZeroBackend()
        return lczero_backend
    else:
        return lczero_backend