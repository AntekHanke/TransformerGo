import chess
import torch
from transformers import BartForConditionalGeneration

from data_processing.chess_tokenizer import ChessTokenizer
from data_processing.data_utils import immutable_boards_to_img
from data_structures.data_structures import ImmutableBoard


class ChessSubgoalGenerator:
    def generate_subgoals(self, immutable_board):
        raise NotImplementedError

class BasicChessSubgoalGenerator(ChessSubgoalGenerator):
    def __init__(self, checkpoint_path_or_model):
        if isinstance(checkpoint_path_or_model, str):
            self.model = BartForConditionalGeneration.from_pretrained(checkpoint_path_or_model)
        else:
            self.model = checkpoint_path_or_model

    def generate_subgoals(self, immutable_board):
        encoded_board = ChessTokenizer.encode_immutable_board(immutable_board) + [
            ChessTokenizer.vocab_to_tokens["<SEP>"]
        ]
        input_tensor = torch.IntTensor([encoded_board]).to(self.model.device)
        outputs = self.model.generate(input_tensor, num_beams=16, max_new_tokens=80).tolist()
        print("".join([ChessTokenizer.tokens_to_vocab[token] for token in outputs[0]]))
        return ChessTokenizer.decode_board(outputs[0])

x = BasicChessSubgoalGenerator("/home/tomek/Research/subgoal_chess_data/generator_k_1/generator_model")
b = chess.Board()
b.push(chess.Move.from_uci("g1h3"))


subgoal = x.generate_subgoals(ImmutableBoard.from_board(chess.Board()))

import matplotlib.pyplot as plt
fig = immutable_boards_to_img([ImmutableBoard.from_board(b), subgoal], ['input', 'target'])

plt.show()
# print(subgoal)

