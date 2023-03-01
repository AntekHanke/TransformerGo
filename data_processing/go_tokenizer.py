from typing import List, Union

import sente
import numpy as np


from data_structures.go_data_structures import GoImmutableBoard


TOKENIZED_BOARD_LENGTH = 361
NON_SPECIAL_TOKENS_START = 10


class MoveDocodingException(Exception):
    pass

class GoTokenizer:
    """Custom tokenizer for go data."""

    TOKENIZED_BOARD_LENGTH = (19*19) + 1 #Whos move
    token_to_boardstate_subarray = {0: [1, 0, 0, 0], 1: [0, 1, 0, 0], 2: [0, 0, 1, 0], 3: [0, 0, 1, 1]}

    @classmethod
    def encode_immutable_board(cls, immutable_board: GoImmutableBoard) -> List[int]:

        whosmove = [(immutable_board.active_player == sente.BLACK)*1]
        boards = immutable_board.boards
        boards_tokens = 3-np.argmax(boards[:,:,[3,2,1,0]], axis=2).flatten() + NON_SPECIAL_TOKENS_START

        board_tokens = whosmove
        board_tokens += boards_tokens.tolist()

        assert (
            len(board_tokens) == cls.TOKENIZED_BOARD_LENGTH
        ), f"The number of tokens encoding the board must be {cls.TOKENIZED_BOARD_LENGTH}, got len(board_tokens) = {len(board_tokens)}"
        return board_tokens

    @classmethod
    def encode_boards(cls, boards: np.ndarray, black_to_move: bool) -> List[int]:

        whosmove = [black_to_move * 1]
        boards_tokens = 3-np.argmax(boards[:,:,[3,2,1,0]], axis=2).flatten() + NON_SPECIAL_TOKENS_START
        board_tokens = whosmove
        board_tokens += boards_tokens.tolist()
        assert (
            len(board_tokens) == cls.TOKENIZED_BOARD_LENGTH
        ), f"The number of tokens encoding the board must be {cls.TOKENIZED_BOARD_LENGTH}, got len(board_tokens) = {len(board_tokens)}"
        return board_tokens


    @classmethod
    def decode_immutable_board(cls, board_tokens: List[int]) -> GoImmutableBoard:
        '''
        Does not support legal_moves feature currently.
        '''
        active_player = sente.BLACK if board_tokens[0] == 1 else sente.WHITE
        boards_list = []
        for token in board_tokens[1:]:
            boards_list.append(cls.token_to_boardstate_subarray[token-NON_SPECIAL_TOKENS_START])
        boards = np.array(boards_list)
        boards = boards.reshape((19,19,4))
        temp_board = GoImmutableBoard.from_all_data(boards,[],active_player,{})
        return temp_board
        # temp_game = temp_board.to_game()
        # immut_board = GoImmutableBoard.from_game(temp_game)
        # return GoImmutableBoard.from_all_data(boards, immut_board.legal_moves)

    @classmethod
    def decode_boards_active_player(cls, board_tokens: List[int]) -> (np.ndarray, bool):
        active_player = (board_tokens[0] == 1)
        boards_list = []
        for token in board_tokens[1:]:
            boards_list.append(cls.token_to_boardstate_subarray[token-NON_SPECIAL_TOKENS_START])
        boards = np.array(boards_list)
        boards = boards.reshape((19,19,4))
        return (boards, active_player)

    @classmethod
    def encode_move(cls, go_move: (int, int, bool)) -> List[int]:
        x, y, is_black = go_move
        return [x*19+y + 400*(1-is_black) + 15]

    @classmethod
    def decode_move(cls, move_token: List[int]) -> (int, int, bool):
        token = move_token[0]-15
        if token>=400:
            is_black = False
            token  -= 400
        else:
            is_black = True

        if token == (19*19+19): #Pass, i.e. (19,19) move
            x = 19
            y = 19
        else:
            y = token%19
            token -= y
            x = token/19
        return (int(x), int(y), is_black)

    # @classmethod
    # def encode(cls, str_or_str_list: Union[List[str], str]) -> List[int]:
    #     if isinstance(str_or_str_list, list):
    #         return [cls.vocab_to_tokens[s] for s in str_or_str_list]
    #     elif isinstance(str_or_str_list, str):
    #         return [cls.vocab_to_tokens[str_or_str_list]]
    #     else:
    #         raise ValueError("str_or_str_list must be a list of strings or a string")
    #
    # @classmethod
    # def decode(cls, tokens):
    #     """General decode method"""
    #     return [cls.tokens_to_vocab[token] for token in tokens]
    #
    # @classmethod
    # def decode_moves(cls, tokens):
    #     """Decode moves"""
    #     return [
    #         chess.Move.from_uci(cls.tokens_to_vocab[token])
    #         for token in tokens
    #         if token not in cls.special_vocab_to_tokens.values()
    #     ]


import pandas as pd

if __name__ == '__main__':

    # generator = SimpleGamesDataGenerator(sgf_files='sgf_directories.txt',save_data_every_n_games=1,p_sample=1,max_games=2,train_eval_split=1,save_path_to_eval_set='eval',save_path_to_train_set='train')
    # generator.create_data()

    np.set_printoptions(threshold=10000)
    aaa = pd.read_pickle("trainsgf_directories.txt_train_part_0.pkl")

    bbb = pd.read_pickle("tokenized_data\\trainval.txt_train_part_2.pkl")
    print(bbb)
    #print(aaa['input_ids'][2].shape)
    #print(np.where(test_arr == 1, test_arr, -1))
    #print(3-np.argmax(test_arr[:,:,[3,2,1,0]], axis=2).flatten())

    GoToken = GoTokenizer()


    # for i in range(len(aaa['labels'])):
        # test_arr = aaa['input_ids'][i]
        # test_arr_2 = aaa['labels'][i]
        # result = GoToken.encode_boards(test_arr, test_arr_2[2])
        # res1, res2 = GoToken.decode_boards_active_player(result)
        #
        # mov_enc = GoToken.encode_move_tuple(test_arr_2)
        # mov_dec = GoToken.decode_move_tuple(mov_enc)
        #
        # assert test_arr_2 == mov_dec
        # assert((test_arr == res1).all())
        # assert(res2 == test_arr_2[2])
        #
        # print(i)

    for i in range(len(bbb['labels'])):
        test_tok = bbb['input_ids'][i]
        test_tok_2 = bbb['labels'][i]
        boards, active = GoToken.decode_boards_active_player(test_tok)
        enc_tok = GoToken.encode_boards(boards, active)

        mov = GoToken.decode_move(test_tok_2)
        enc_mov = GoToken.encode_move(mov)


        # res1, res2 = GoToken.decode_boards_active_player(result)

        # mov_enc = GoToken.encode_move_tuple(test_arr_2)
        # mov_dec = GoToken.decode_move_tuple(mov_enc)

        assert enc_tok == test_tok
        assert enc_mov == test_tok_2
        # assert((test_arr == res1).all())
        # assert(res2 == test_arr_2[2])

        #print(i)