from ctypes import Union
from typing import List

import chess
import torch
from transformers import BartForConditionalGeneration

from data_processing.chess_tokenizer import ChessTokenizer
from data_structures.data_structures import ImmutableBoard


class ChessPolicy:
    """Core class for policy"""

    def get_best_moves(self, immutable_board: ImmutableBoard, num_return_sequences: int) -> List[chess.Move]:
        raise NotImplementedError


class BasicChessPolicy(ChessPolicy):
    """Basic policy based on generation from the model"""

    # TODO: checkpoint_path_or_model typing
    def __init__(self, checkpoint_path_or_model) -> None:
        if isinstance(checkpoint_path_or_model, str):
            self.model = BartForConditionalGeneration.from_pretrained(checkpoint_path_or_model)
        else:
            self.model = checkpoint_path_or_model

    def get_best_moves(self, immutable_board: ImmutableBoard, num_return_sequences: int) -> List[chess.Move]:
        encoded_board: List[int] = ChessTokenizer.encode_immutable_board(immutable_board) + [
            ChessTokenizer.vocab_to_tokens["<SEP>"]
        ]
        input_tensor: torch.Tensor = torch.IntTensor([encoded_board]).to(self.model.device)
        outputs: List[List[int]] = self.model.generate(
            input_tensor, num_beams=16, max_new_tokens=4, num_return_sequences=num_return_sequences
        ).tolist()
        moves: List[chess.Move] = [ChessTokenizer.decode_move(output) for output in outputs]
        return moves
