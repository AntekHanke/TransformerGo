import time
from typing import List, Tuple, Dict, Union

import torch
from transformers import BartForConditionalGeneration

from configures.global_config import TOKENIZED_BOARD_LEN
from data_processing.chess_tokenizer import ChessTokenizer
from data_structures.data_structures import ImmutableBoard
from metric_logging import log_value_to_average, log_value_to_accumulate


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


class ChessSubgoalGenerator:
    def __init__(self, checkpoint_path_or_model: Union[str, BartForConditionalGeneration]):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Device used for evaluation subgoal generator:", device)
        if isinstance(checkpoint_path_or_model, str):
            self.model = BartForConditionalGeneration.from_pretrained(checkpoint_path_or_model)
        else:
            self.model = checkpoint_path_or_model
        self.model.to(device)
        self.model.eval()
        self.memory = {}

    def is_in_memory(self, input_board: ImmutableBoard) -> bool:
        return input_board in self.memory

    def generate_subgoals(
        self,
        input_boards: List[ImmutableBoard],
        generator_num_beams: int,
        generator_num_subgoals: int,
        subgoal_distance_k: int,
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
        subgoal_distance_k: int,
        **subgoal_generation_kwargs,
    ) -> List[List[ImmutableBoard]]:
        assert (
            isinstance(subgoal_distance_k, int) and 10 > subgoal_distance_k > 0
        ), "subgoal_distance_k must be an integer between 1 and 9"

        encoded_boards = [
            [5000 + subgoal_distance_k]
            + ChessTokenizer.encode_immutable_board(input_board)
            + [ChessTokenizer.vocab_to_tokens["<SEP>"]]
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
        all_subgoals = [
            [ChessTokenizer.decode_board(sequence) for sequence in subgoal_out]
            for subgoal_out in chunks(outputs, generator_num_subgoals)
        ]
        for input_board, subgoals in zip(input_boards, all_subgoals):
            self.memory[input_board] = subgoals
        return all_subgoals
