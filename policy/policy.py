import chess
import torch
from transformers import AutoModel, BartForConditionalGeneration, BartModel

from data_processing.chess_tokenizer import ChessTokenizer, board_to_board_state


class Policy:
    def __init__(self, model_checkpoint_path):
        self.model =  BartForConditionalGeneration.from_pretrained(model_checkpoint_path)

    def get_best_move(self, board):
        encoded_board = ChessTokenizer.encode_board(board)
        input_tensor = torch.IntTensor([encoded_board])
        outputs = self.model.generate(input_tensor, num_beams=32, num_return_sequences=1, max_new_tokens=4).tolist()
        #
        # print(outputs)
        # for s in outputs[0]:
        #     print(f'char = {ChessTokenizer.tokens_to_vocab[s]}')
        #
        # print(ChessTokenizer.decode_move(outputs[0][1:4]))
        return ChessTokenizer.decode_move(outputs[0][1:4])
        # print(ChessTokenizer.decode_move(outputs[1][1:4]))



# ufik = Policy("/home/tomek/Research/subgoal_chess_data/local_policy")
# board = chess.Board()
# print(board.fen())
#
#
#
# ufik.get_best_moves(board)