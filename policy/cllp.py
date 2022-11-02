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

    def get_path(self, input_immutable_board, target_immutable_board):
        model_input = (
            ChessTokenizer.encode_immutable_board(input_immutable_board)
            + [ChessTokenizer.special_vocab_to_tokens["<SEP>"]]
            + [
                ChessTokenizer.special_vocab_to_tokens["<SEP>"]
            ]  # This is not mistake, it is the correction for the bug in train data
            + ChessTokenizer.encode_immutable_board(target_immutable_board)
        )
        input_tensor = torch.IntTensor([model_input]).to(self.model.device)
        output = self.model.generate(input_tensor, max_length=40).tolist()
        return ChessTokenizer.decode_uci_moves(output[0])

    def get_batch_path(self, queries_list):
        inputs_tokenized = []
        targets_tokenized = []
        moves_batch = []

        for input_immutable_board, target_immutable_board in queries_list:
            inputs_tokenized.append(
                ChessTokenizer.encode_immutable_board(input_immutable_board)
                + [ChessTokenizer.special_vocab_to_tokens["<SEP>"]]
                + [
                    ChessTokenizer.special_vocab_to_tokens["<SEP>"]
                ]  # This is not mistake, it is the correction for the bug in train data
                + ChessTokenizer.encode_immutable_board(target_immutable_board)
            )

        input_tensor = torch.IntTensor(inputs_tokenized).to(self.model.device)
        output = self.model.generate(input_tensor, max_length=40).tolist()

        for out in output:
            try:
                moves_batch.append(ChessTokenizer.decode_uci_moves(out))
            except:
                moves_batch.append(None)

        return moves_batch
