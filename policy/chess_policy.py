from ctypes import Union
from typing import List

import chess
import numpy as np
import torch
from transformers import BartForConditionalGeneration

from configures.global_config import MAX_NEW_TOKENS_FOR_POLICY
from data_processing.chess_tokenizer import ChessTokenizer
from data_structures.data_structures import ImmutableBoard
from lczero.lczero_backend import LCZeroBackend, get_lczero_backend
from utils.chess960_conversion import chess960_to_standard


class ChessPolicy:
    """Core class for pgn_policy"""

    def get_best_moves(self, immutable_board: ImmutableBoard, num_return_sequences: int) -> List[chess.Move]:
        raise NotImplementedError


class BasicChessPolicy(ChessPolicy):
    """Basic pgn_policy based on generation from the model"""

    # TODO: checkpoint_path_or_model typing
    def __init__(self, checkpoint_path_or_model) -> None:
        if isinstance(checkpoint_path_or_model, str):
            self.model = BartForConditionalGeneration.from_pretrained(checkpoint_path_or_model)
        else:
            self.model = checkpoint_path_or_model

    def get_best_moves(
        self,
        immutable_board: ImmutableBoard,
        num_return_sequences: int = 8,
        num_beams: int = 16,
        return_probs: bool = False,
        do_sample: bool = False,
    ) -> List[chess.Move]:
        encoded_board: List[int] = ChessTokenizer.encode_immutable_board(immutable_board) + [
            ChessTokenizer.vocab_to_tokens["<SEP>"]
        ]
        input_tensor: torch.Tensor = torch.IntTensor([encoded_board]).to(self.model.device)
        outputs: List[List[int]] = self.model.generate(
            inputs=input_tensor,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            max_new_tokens=MAX_NEW_TOKENS_FOR_POLICY,
            output_scores=True,
            return_dict_in_generate=True,
            do_sample=False,
        )

        sequence = outputs.sequences.tolist()
        scores = outputs.scores[0]

        moves: List[chess.Move] = []
        moves_ids: List[int] = []
        if not do_sample:
            for output in sequence:
                output = [x for x in output if x not in ChessTokenizer.special_vocab_to_tokens.values()]
                moves.append(ChessTokenizer.decode_move(output))
                moves_ids.append(output)
        else:
            probs = np.exp(scores[0].tolist())
            probs = probs / np.sum(probs)
            moves_ids = np.random.choice(list(range(4600)), num_return_sequences, p=probs, replace=False)
            moves = [ChessTokenizer.decode_move([x]) for x in moves_ids]

        print(f"Moves = {[str(move) for move in moves]}")

        board = immutable_board.to_board()
        converted_moves = [chess960_to_standard(move, board) for move in moves]

        if return_probs:
            logits = [scores[0, move_id].tolist() for move_id in moves_ids]
            return converted_moves, np.exp(logits)
        else:
            return converted_moves

    def find_move_probability(self, immutable_board: ImmutableBoard, move_str: chess.Move) -> float:
        encoded_board: List[int] = ChessTokenizer.encode_immutable_board(immutable_board) + [
            ChessTokenizer.vocab_to_tokens["<SEP>"]
        ]
        input_tensor: torch.Tensor = torch.IntTensor([encoded_board]).to(self.model.device)
        outputs = self.model.generate(
            inputs=input_tensor,
            output_scores=True,
            max_new_tokens=2,
            return_dict_in_generate=True,
        )
        sequence = outputs.sequences.tolist()
        # move = ChessTokenizer.decode_move(sequence)
        scores = outputs.scores[0]
        num = scores[0, 100].tolist()
        print(num)


class LCZeroPolicy(ChessPolicy):
    def __init__(self):
        self.lczero_backend = get_lczero_backend()

    def get_best_moves(
        self,
        immutable_board: ImmutableBoard,
        num_return_sequences: int = 8,
        num_beams: int = None,
        return_probs: bool = False,
        do_sample: bool = False,
    ):
        p_distribution = self.lczero_backend.get_policy_distribution(immutable_board)[:num_return_sequences]
        print(p_distribution)
        probs = []
        moves = []
        for move in p_distribution:
            moves.append(chess.Move.from_uci(move[0]))
            probs.append(move[1])
        if return_probs:
            return moves, probs
        else:
            return moves

    def get_path_probability(
        self, immutable_board: ImmutableBoard, path: List[chess.Move], log_prob: bool = True
    ) -> float:
        probs = self.lczero_backend.get_path_probabilities(immutable_board, path)
        if log_prob:
            return sum(np.log(probs))
        else:
            return np.prod(probs)

    def get_path_probabilities(self, immutable_board: ImmutableBoard, path: List[chess.Move]) -> float:
        board = immutable_board.to_board()
        probs = []
        for move in path:
            moves_distribution = self.lczero_backend.policy_distribution_dict(ImmutableBoard.from_board(board))
            if move.uci() in moves_distribution:
                probs.append(moves_distribution[move.uci()])
            else:
                probs.append(0)
            board.push(move)
        return probs
