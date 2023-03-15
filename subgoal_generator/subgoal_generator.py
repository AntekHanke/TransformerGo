import time
from typing import List, Tuple, Dict, Union

import torch
from transformers import BartForConditionalGeneration

from configures.global_config import TOKENIZED_BOARD_LEN
from data_processing.chess_tokenizer import ChessTokenizer
from data_structures.data_structures import ImmutableBoard
from metric_logging import log_object, log_value_to_average, log_value_to_accumulate


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


class ChessSubgoalGenerator:
    def __init__(self, checkpoint_path_or_model: Union[str, BartForConditionalGeneration]):
        if isinstance(checkpoint_path_or_model, str):
            self.model = BartForConditionalGeneration.from_pretrained(checkpoint_path_or_model)
        else:
            self.model = checkpoint_path_or_model
        self.memory = {}

    def is_in_memory(self, input_board: ImmutableBoard) -> bool:
        return input_board in self.memory

    def generate_subgoals(
        self,
        input_boards: List[ImmutableBoard],
        generator_num_beams: int,
        generator_num_subgoals: int,
        **subgoal_generation_kwargs,
    ) -> List[Tuple[List[ImmutableBoard], Dict]]:
        raise NotImplementedError

    def generate_use_memory(self, input_board: ImmutableBoard):
        if input_board in self.memory:
            return self.memory[input_board]
        else:
            raise ValueError("Board not in memory")


class BasicChessSubgoalGenerator(ChessSubgoalGenerator):
    def generate_subgoals(
        self,
        input_boards: List[ImmutableBoard],
        generator_num_beams: int,
        generator_num_subgoals: int,
        **subgoal_generation_kwargs,
    ) -> List[List[ImmutableBoard]]:

        encoded_boards = [
            ChessTokenizer.encode_immutable_board(input_board) + [ChessTokenizer.vocab_to_tokens["<SEP>"]]
            for input_board in input_boards
        ]
        input_tensor = torch.IntTensor(encoded_boards).to(self.model.device)

        time_start = time.time()
        outputs = self.model.generate(
            input_tensor,
            max_new_tokens=TOKENIZED_BOARD_LEN + 1,
            num_beams=generator_num_beams,
            num_return_sequences=generator_num_subgoals,
            **subgoal_generation_kwargs,
        ).tolist()
        log_value_to_average("subgoal_generation_time_avg", time.time() - time_start)
        log_value_to_accumulate("subgoal_generation_time_total", time.time() - time_start)
        subgoals = [
            [ChessTokenizer.decode_board(sequence) for sequence in subgoal_out]
            for subgoal_out in chunks(outputs, generator_num_subgoals)
        ]
        for input_board, subgoals in zip(input_boards, subgoals):
            self.memory[input_board] = subgoals
        return subgoals


class AdaChessSubgoalGenerator(ChessSubgoalGenerator):
    def __init__(self, checkpoint_path_or_model, subgoal_distance: int = 1) -> None:
        super().__init__(checkpoint_path_or_model)
        self.subgoal_distance = subgoal_distance
        assert (
            isinstance(self.subgoal_distance, int) and 10 > self.subgoal_distance > 0
        ), "Subgoal distance must be an int in range [1, 10]"

    def generate_subgoals(
        self, input_board: ImmutableBoard, time_info: bool = False, **subgoal_generation_kwargs
    ) -> List[ImmutableBoard]:
        encoded_board = (
            [self.subgoal_distance]
            + ChessTokenizer.encode_immutable_board(input_board)
            + [ChessTokenizer.vocab_to_tokens["<SEP>"]]
        )
        input_tensor = torch.IntTensor([encoded_board]).to(self.model.device)
        time_start = time.time()
        outputs = self.model.generate(input_tensor, max_new_tokens=80, **subgoal_generation_kwargs).tolist()
        if time_info:
            print(f"Subgoal generation time: {time.time() - time_start}")
        subgoals = []
        for sequence in outputs:
            subgoals.append(ChessTokenizer.decode_board(sequence))

        # TODO: Is this necessary?
        # subgoals = list({subgoal for subgoal in subgoals if subgoal.board != input_board.board})
        return subgoals
