import chess
import torch
from transformers import AutoModel, BartForConditionalGeneration, BartModel

from data_processing.chess_tokenizer import ChessTokenizer, board_to_immutable_board


class Policy:
    def __init__(self, checkpoint_path_or_model):
        if isinstance(checkpoint_path_or_model, str):
            self.model =  BartForConditionalGeneration.from_pretrained(checkpoint_path_or_model)
        else:
            self.model = checkpoint_path_or_model

    def get_best_move(self, board):
        encoded_board = ChessTokenizer.encode_immutable_board(board)
        input_tensor = torch.IntTensor([encoded_board]).to(self.model.device)
        outputs = self.model.generate(input_tensor).tolist()
        return ChessTokenizer.decode_move(outputs[0][1:4])


# ufik = Policy("/home/tomek/Research/subgoal_chess_data/local_policy")
# board = chess.Board()
# print(board.fen())
#
#
#
# ufik.get_best_moves(board)