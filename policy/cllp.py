from typing import List, Tuple

import torch
import time
from transformers import BartForConditionalGeneration

from data_processing.chess_tokenizer import ChessTokenizer
from data_structures.data_structures import ImmutableBoard


class CLLP:
    """Basic pgn_policy based on generation from the model"""

    def __init__(self, checkpoint_path_or_model):
        if isinstance(checkpoint_path_or_model, str):
            self.model = BartForConditionalGeneration.from_pretrained(checkpoint_path_or_model)
        else:
            self.model = checkpoint_path_or_model

    @staticmethod
    def input_and_target_to_list_of_tokens(
        input_immutable_board: ImmutableBoard, target_immutable_board: ImmutableBoard
    ):
        return (
            ChessTokenizer.encode_immutable_board(input_immutable_board)
            + [ChessTokenizer.special_vocab_to_tokens["<SEP>"]]
            + [
                ChessTokenizer.special_vocab_to_tokens["<SEP>"]
            ]  # This is not mistake, it is the correction for the bug in train data
            + ChessTokenizer.encode_immutable_board(target_immutable_board)
        )

    def generate_moves_batch_from_model(self, input_tokens, num_beams, num_return_sequences):
        input_tensor = torch.IntTensor(input_tokens).to(self.model.device)
        output = self.model.generate(
            input_tensor,
            max_length=40,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
        )
        output = output.tolist()
        moves_batch = []
        moves_for_one_query = []
        for i, out in enumerate(output):
            moves_for_one_query.append(ChessTokenizer.decode_moves(out))
            if (i + 1) % num_return_sequences == 0:
                moves_batch.append(moves_for_one_query)
                moves_for_one_query = []
        return moves_batch

    def get_path(
        self,
        input_immutable_board: ImmutableBoard,
        target_immutable_board: ImmutableBoard,
        num_beams,
        num_return_sequences,
    ):
        model_input = self.input_and_target_to_list_of_tokens(input_immutable_board, target_immutable_board)
        moves_batch = self.generate_moves_batch_from_model([model_input], num_beams, num_return_sequences)
        return moves_batch[0]

    def get_paths_batch(
        self, queries_list: List[Tuple[ImmutableBoard, ImmutableBoard]], num_beams, num_return_sequences
    ):

        time_cllp = time.time()
        inputs_tokenized = []
        for input_immutable_board, target_immutable_board in queries_list:
            inputs_tokenized.append(
                self.input_and_target_to_list_of_tokens(input_immutable_board, target_immutable_board)
            )
        return self.generate_moves_batch_from_model(inputs_tokenized, num_beams, num_return_sequences), {
            "cllp_time": time.time() - time_cllp
        }
