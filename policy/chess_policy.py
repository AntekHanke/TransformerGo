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

    def get_best_moves(self, immutable_board: ImmutableBoard, num_return_sequences: int = 1) -> List[chess.Move]:
        encoded_board: List[int] = ChessTokenizer.encode_immutable_board(immutable_board) + [
            ChessTokenizer.vocab_to_tokens["<SEP>"]
        ]
        input_tensor: torch.Tensor = torch.IntTensor([encoded_board]).to(self.model.device)
        outputs: List[List[int]] = self.model.generate(
            inputs=input_tensor, num_return_sequences=num_return_sequences, max_new_tokens=MAX_NEW_TOKENS_FOR_POLICY
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
        num = scores[0,100].tolist()
        print(num)

    # def find_move_probability(self, immutable_board: ImmutableBoard, move: chess.Move):
    #     encoded_board: List[int] = ChessTokenizer.encode_immutable_board(immutable_board) + [
    #         ChessTokenizer.vocab_to_tokens["<SEP>"]
    #     ]
    #     encoded_move: List[int] = ChessTokenizer.encode_move(move)
    #     input_tensor: torch.Tensor = torch.IntTensor([encoded_board]).to(self.model.device)
    #     forced_ids = [[0, ChessTokenizer.special_vocab_to_tokens["<EOS>"]]]
    #     for token_num, token in enumerate(encoded_move):
    #         forced_ids.append([token_num + 1, token])
    #     outputs = self.model.generate(
    #         inputs=input_tensor,
    #         output_scores=True,
    #         max_new_tokens=5,
    #         return_dict_in_generate=True,
    #         forced_decoder_ids=forced_ids,
    #     )
    #     sequence = outputs.sequences.tolist()[0]
    #     move = ChessTokenizer.decode_move(sequence)
    #     scores = outputs.scores
    #     logits_sum = 0
    #     for num_token in forced_ids[1:]:
    #         num, token = num_token
    #         token_scores = scores[num - 1].tolist()[0]
    #         print(f"score = {token_scores[token+1]}")
    #         logits_sum += token_scores[token]
    #     return move, logits_sum
