from typing import Optional

from assets.lczero.lczero_weights_path import DEFAULT_LC_ZERO_WEIGHTS_PATH
from utils.detect_local_machine import is_local_machine

# global lczero_backend
lczero_backend = None

class LCZeroBackend:
    def __init__(self, weights_class, backend_class, game_state_class, weights_path: Optional[str] = None):
        if weights_path is None:
            weights_path = DEFAULT_LC_ZERO_WEIGHTS_PATH
        self.weights = weights_class(weights_path)
        self.backend = backend_class(weights=self.weights)
        self.game_state_class = game_state_class

    def evaluate_immutable_board(self, immutable_board):
        game_state = self.game_state_class(fen=immutable_board.fen())
        input = game_state.as_input(self.backend)
        output = self.backend.evaluate(input)[0]
        return output.q()

    def get_policy_distribution(self, immutable_board):
        game_state = self.game_state_class(fen=immutable_board.fen())
        input = game_state.as_input(self.backend)
        output = self.backend.evaluate(input)[0]
        moves = list(zip(game_state.moves(), output.p_softmax(*game_state.policy_indices())))
        sorted_moves = sorted(moves, key=lambda x: x[1], reverse=True)
        return sorted_moves

    def policy_distribution_dict(self, immutable_board):
        sorted_moves = self.get_policy_distribution(immutable_board)
        return {move[0]: move[1] for move in sorted_moves}

def get_lczero_backend(weights_path=None):
    """Generates a new LCZero backend if one doesn't exist, otherwise returns the existing one."""
    global lczero_backend
    if lczero_backend is None:
        if is_local_machine():
            from lczero_backend_local.lczero_local import lc_zero_local_backend
            lczero_backend = lc_zero_local_backend
        else:
            from lczero_backend_athena.lczero_athena import lc_zero_athena_backend
            lczero_backend = lc_zero_athena_backend
        return lczero_backend
    else:
        return lczero_backend