import chess
import torch
from transformers import BartForConditionalGeneration

from data_processing.chess_tokenizer import ChessTokenizer


class ChessPolicy:
    """Core class for policy"""

    def get_best_move(self, immutable_board):
        raise NotImplementedError


class BasicChessPolicy(ChessPolicy):
    """Basic policy based on generation from the model"""
    def __init__(self, checkpoint_path_or_model):
        if isinstance(checkpoint_path_or_model, str):
            self.model = BartForConditionalGeneration.from_pretrained(checkpoint_path_or_model)
        else:
            self.model = checkpoint_path_or_model

    def get_best_move(self, immutable_board):
        encoded_board = ChessTokenizer.encode_immutable_board(immutable_board) + [
            ChessTokenizer.vocab_to_tokens["<SEP>"]
        ]
        input_tensor = torch.IntTensor([encoded_board]).to(self.model.device)
        outputs = self.model.generate(input_tensor, num_beams=16, max_new_tokens=4).tolist()
        return ChessTokenizer.decode_move(outputs[0])
