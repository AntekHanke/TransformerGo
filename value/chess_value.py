from data_structures.data_structures import ImmutableBoard
from lczero.lczero_backend import LCZeroBackend, get_lczero_backend


class ChessValue:
    def evaluate_immutable_board(self, immutable_board: ImmutableBoard) -> float:
        raise NotImplementedError


class LCZeroValue(ChessValue):
    def __init__(self) -> None:
        self.lczero_backend = get_lczero_backend()

    @staticmethod
    def absolute_v(player, v):
        if player == "w":
            return v
        elif player == "b":
            return -v

    def evaluate_immutable_board(self, immutable_board: ImmutableBoard) -> float:
        return self.absolute_v(
            immutable_board.active_player, self.lczero_backend.evaluate_immutable_board(immutable_board)
        )
