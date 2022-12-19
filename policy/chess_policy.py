from ctypes import Union
from typing import List

import chess
import torch
from transformers import BartForConditionalGeneration

from configures.global_config import MAX_NEW_TOKENS_FOR_POLICY
from data_processing.chess_tokenizer import ChessTokenizer
from data_structures.data_structures import ImmutableBoard


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
        self, immutable_board: ImmutableBoard, num_return_sequences: int = 1, num_beams: int = 16
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
            do_sample=False,
        ).tolist()
        print(f"decoded = {ChessTokenizer.decode(outputs[0])}")
        moves: List[chess.Move] = []
        for output in outputs:
            output = [x for x in output if x not in ChessTokenizer.special_vocab_to_tokens.values()]
            moves.append(ChessTokenizer.decode_move(output))
        return moves

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
