import random

import chess
import torch
from transformers import BartForConditionalGeneration

from data_processing.chess_tokenizer import ChessTokenizer


class ChessPolicy:
    """Core class for policy"""

    def get_best_move(self, immutable_board):
        raise NotImplementedError


class CLLP(ChessPolicy):
    """Basic policy based on generation from the model"""

    def __init__(self, checkpoint_path_or_model):
        if isinstance(checkpoint_path_or_model, str):
            self.model = BartForConditionalGeneration.from_pretrained(checkpoint_path_or_model)
        else:
            self.model = checkpoint_path_or_model

    def get_path(self, input_immutable_board, subgoal_immutable_board):
        model_input = (
            ChessTokenizer.encode_immutable_board(input_immutable_board)
            + [ChessTokenizer.special_vocab_to_tokens["<SEP>"]]
            + [
                ChessTokenizer.special_vocab_to_tokens["<SEP>"]
            ]  # This is not mistake, it is the correction for the bug in train data
            + ChessTokenizer.encode_immutable_board(subgoal_immutable_board)
        )
        print(len(model_input))
        input_tensor = torch.IntTensor([model_input]).to(self.model.device)
        output = self.model.generate(input_tensor).tolist()
        return ChessTokenizer.decode(output[0])
        # return ChessTokenizer.decode_move(output[0])


class CLLP_am(ChessPolicy):
    """Basic policy based on generation from the model"""

    def __init__(self, checkpoint_path_or_model):
        if isinstance(checkpoint_path_or_model, str):
            self.model = BartForConditionalGeneration.from_pretrained(checkpoint_path_or_model)
        else:
            self.model = checkpoint_path_or_model

    def get_path(self, input_immutable_board, subgoal_immutable_board):
        model_input = (
            ChessTokenizer.encode_immutable_board(input_immutable_board)
            + [ChessTokenizer.special_vocab_to_tokens["<SEP>"]]
            + [
                ChessTokenizer.special_vocab_to_tokens["<SEP>"]
            ]  # This is not mistake, it is the correction for the bug in train data
            + ChessTokenizer.encode_immutable_board(subgoal_immutable_board)
        )
        input_tensor = torch.IntTensor([model_input]).to(self.model.device)
        output = self.model.generate(input_tensor, max_length=40).tolist()
        return ChessTokenizer.decode(output[0])
