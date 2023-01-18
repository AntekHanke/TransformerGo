import time
from typing import List

import chess
import torch
from transformers import BartForConditionalGeneration

from data_processing.chess_tokenizer import ChessTokenizer
from data_structures.data_structures import ImmutableBoard


class ChessSubgoalGenerator:
    def generate_subgoals(
        self, immutable_board: ImmutableBoard, n_subgoals: int, **subgoal_generation_kwargs
    ) -> List[ImmutableBoard]:
        raise NotImplementedError


class BasicChessSubgoalGenerator(ChessSubgoalGenerator):
    def __init__(self, checkpoint_path_or_model):
        if isinstance(checkpoint_path_or_model, str):
            self.model = BartForConditionalGeneration.from_pretrained(checkpoint_path_or_model)
        else:
            self.model = checkpoint_path_or_model

    def generate_subgoals(
        self, input_board: ImmutableBoard, time_info: bool = False, **subgoal_generation_kwargs
    ) -> List[ImmutableBoard]:
        encoded_board = ChessTokenizer.encode_immutable_board(input_board) + [ChessTokenizer.vocab_to_tokens["<SEP>"]]
        input_tensor = torch.IntTensor([encoded_board]).to(self.model.device)
        time_start = time.time()
        outputs = self.model.generate(input_tensor, max_new_tokens=80, **subgoal_generation_kwargs).tolist()
        if time_info:
            print(f"Subgoal generation time: {time.time() - time_start}")
        subgoals = []
        for sequence in outputs:
            subgoals.append(ChessTokenizer.decode_board(sequence))

        # subgoals = list({subgoal for subgoal in subgoals if subgoal.board != input_board.board})
        return subgoals
