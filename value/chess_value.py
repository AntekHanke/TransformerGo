from data_structures.data_structures import ImmutableBoard
from lczero.lczero_backend import LCZeroBackend, get_lczero_backend


class ChessValue:
    def evaluate_immutable_board(self, immutable_board: ImmutableBoard) -> float:
        raise NotImplementedError

class LCZeroValue(ChessValue):
    def __init__(self, lczero_backend: LCZeroBackend = None) -> None:
        self.lczero_backend = get_lczero_backend()

    def evaluate_immutable_board(self):
        return self.lczero_backend.evaluate_immutable_board()