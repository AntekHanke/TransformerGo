from typing import Optional, Dict, List, Tuple, Any

import numpy as np
import sente
import torch
from sente import stone

from board import Board
from data_structures.go_data_structures import GoImmutableBoard
from load_model import load_model
from features import Features

BOARD_SIZE = 19
DEFAULT_WEIGHTS_PATH = "/home/malgorzatarog/ProjectData/subgoal_chess_data/model.ckpt"

RULES = {
    "koRule": "KO_POSITIONAL",
    "scoringRule": "SCORING_AREA",
    "taxRule": "TAX_NONE",
    "multiStoneSuicideLegal": True,
    "hasButton": False,
    "encorePhase": 0,
    "passWouldEndPhase": False,
    "whiteKomi": 7.5,
    "asymPowersOfTwo": 0.0,
}


class KataGoBackend:
    def __init__(
        self,
        weights_path: Optional[str] = None,
        use_swa: bool = False,
        board_size: Optional[int] = None,
    ) -> None:
        if weights_path is None:
            weights_path = DEFAULT_WEIGHTS_PATH
        self.board_size = BOARD_SIZE if board_size is None else board_size
        model, swa_model, _ = load_model(
            weights_path, use_swa, device="cpu", pos_len=self.board_size, verbose=False
        )
        model.eval()
        self.model_config = model.config
        if swa_model is not None:
            model = swa_model.module
            model.eval()
        self.model = model
        self.features = Features(self.model_config, self.board_size)

    def evaluate_immutable_board(
        self, immutable_board: GoImmutableBoard, move_history: List[sente.Move] = None
    ) -> Dict[str, float]:
        outputs = self.process_board(immutable_board, move_history)["value"]
        value = {"win": outputs[0], "loss": outputs[1], "noresult": outputs[2]}
        return value

    def get_policy_distribution(
        self, immutable_board: GoImmutableBoard, move_history: List[sente.Move] = None
    ) -> List[Tuple[Optional[Tuple[int, int, stone]], float]]:
        outputs = self.process_board(immutable_board, move_history)
        moves_and_probs = outputs["moves_and_probs"]
        moves_and_probs = [
            (self.katago_to_sente_move(move[0], immutable_board.active_player), move[1])
            for move in moves_and_probs
        ]
        return moves_and_probs

    def policy_distribution_dict(
        self, immutable_board: GoImmutableBoard, move_history: List[sente.Move] = None
    ) -> Dict[Optional[Tuple[int, int, stone]], float]:
        moves_and_probs = self.get_policy_distribution(immutable_board, move_history)
        return {move[0]: move[1] for move in moves_and_probs}

    def process_board(
        self, immutable_board: GoImmutableBoard, move_history: List[sente.Move]
    ) -> Dict[str, Any]:
        if move_history is None:
            move_history = []
        board = self.immutable_board_to_katago_board(immutable_board)
        board_history = self.get_board_history(immutable_board, move_history)
        move_history = [self.sente_to_katago_move(move) for move in move_history]
        with torch.no_grad():
            # self.model.eval()
            bin_input_data = np.zeros(
                shape=[1] + self.model.bin_input_shape, dtype=np.float32
            )
            global_input_data = np.zeros(
                shape=[1] + self.model.global_input_shape, dtype=np.float32
            )
            pla = board.pla
            opp = Board.get_opp(pla)
            move_idx = len(move_history)
            bin_input_data = np.transpose(bin_input_data, axes=(0, 2, 3, 1))
            bin_input_data = bin_input_data.reshape(
                [1, self.board_size * self.board_size, -1]
            )
            self.features.fill_row_features(
                board,
                pla,
                opp,
                board_history,
                move_history,
                move_idx,
                RULES,
                bin_input_data,
                global_input_data,
                idx=0,
            )
            bin_input_data = bin_input_data.reshape(
                [1, self.board_size, self.board_size, -1]
            )
            bin_input_data = np.transpose(bin_input_data, axes=(0, 3, 1, 2))

            model_outputs = self.model(
                torch.tensor(bin_input_data, dtype=torch.float32),
                torch.tensor(global_input_data, dtype=torch.float32),
            )
            outputs = self.model.postprocess_output(model_outputs)
            policy_logits = outputs[0][0][0]
            value_logits = outputs[0][1][0]

            policy = (
                torch.nn.functional.softmax(policy_logits[0, :], dim=0).cpu().numpy()
            )
            value = torch.nn.functional.softmax(value_logits, dim=0).cpu().numpy()

        moves_and_probs = []
        for i in range(len(policy)):
            move = self.features.tensor_pos_to_loc(i, board)
            if i == len(policy) - 1:
                moves_and_probs.append((Board.PASS_LOC, policy[i]))
            elif board.would_be_legal(board.pla, move):
                moves_and_probs.append((move, policy[i]))
        moves_and_probs = sorted(
            moves_and_probs, key=lambda moveandprob: moveandprob[1], reverse=True
        )

        return {
            "policy": policy,
            "moves_and_probs": moves_and_probs,
            "value": value,
        }

    def immutable_board_to_katago_board(
        self, immutable_board: GoImmutableBoard
    ) -> Board:
        assert (
            immutable_board.boards.shape[0] == self.board_size
            and immutable_board.boards.shape[1] == self.board_size
        ), f"Model board size {self.board_size} does not match input board size {immutable_board.boards.size[0:2]}x"
        board = Board(size=self.board_size)
        for row in range(self.board_size):
            for col in range(self.board_size):
                if immutable_board.boards[row][col][0] == 1:
                    board.add_unsafe(Board.BLACK, board.loc(row, col))
                elif immutable_board.boards[row][col][1] == 1:
                    board.add_unsafe(Board.WHITE, board.loc(row, col))
        if immutable_board.active_player == sente.stone.BLACK:
            board.set_pla(Board.BLACK)
        elif immutable_board.active_player == sente.stone.WHITE:
            board.set_pla(Board.WHITE)
        else:
            raise ValueError(f"Unknown player {immutable_board.active_player}")
        return board

    def sente_to_katago_move(self, move: sente.Move) -> int:
        return Board.loc_static(move.get_x(), move.get_y(), self.board_size)

    def katago_to_sente_move(
        self, move: int, player: int
    ) -> Tuple[Optional[int], Optional[int], sente.stone]:
        if player == Board.BLACK:
            player = sente.BLACK
        elif player == Board.WHITE:
            player = sente.WHITE
        else:
            raise ValueError(f"Unknown player {player}")
        x_coordinate = (move % (self.board_size + 1))
        y_coordinate = (move // (self.board_size + 1))
        if x_coordinate == 0 and y_coordinate == 0:
            return None, None, player
        else:
            return x_coordinate, y_coordinate, player

    def get_board_history(
        self, immutable_board: GoImmutableBoard, move_history: List[sente.Move]
    ) -> List[Board]:
        board_history = [immutable_board]
        for move in move_history:
            immutable_board = immutable_board.act(move)
            board_history.append(immutable_board)
        board_history = [
            self.immutable_board_to_katago_board(board) for board in board_history
        ]
        return board_history
