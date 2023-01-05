from lczero.backends import Weights, Backend, GameState


class LCZeroBackend:
    def __init__(self):
        try:
            self.weights = Weights()
            self.backend = Backend(weights=self.weights)
        except:
            print(
                "Weights not found. Please download them from https://lczero.org/networks/ and place them in the same folder as this script."
            )

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
