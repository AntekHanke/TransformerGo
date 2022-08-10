import chess
import torch
from transformers import AutoModel

from data_processing.chess_tokenizer import ChessTokenizer, board_to_board_state


class Policy:
    def __init__(self, model_checkpoint_path):
        self.model = AutoModel.from_pretrained(model_checkpoint_path)

    def get_best_move(self, board):
        encoded_board = ChessTokenizer.encode_board(board_to_board_state(board))
        outputs = self.model.generate(torch.IntTensor([[10, 20, 30]]), num_beams=4, num_return_sequences=2)


board = chess.Board()
print(board.fen())