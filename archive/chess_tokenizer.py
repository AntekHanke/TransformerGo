#
#
# class ChessTokenizer:
#
#     pieces = [' ', 'P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k', '/', '.']
#     squares = list(range(1,64))
#     castlings = ['KQkq', 'KQk', 'KQ', 'K', 'Qkq', 'Qk', 'Q', 'kq', 'k', 'q', 'Kkq', 'Kq', 'Kk', 'Qq',
#                  'KQq', '-']
#     players = ['w', 'b']
#     special_tokes = ['<PAD>', '<SEP>', '<EOS>']
#     vocab =  special_tokes + pieces + squares + players + castlings
#     vocab_to_tokens = {symbol: i for i, symbol in enumerate(vocab)}
#     tokes_to_vocab = {i: symbol for i, symbol in enumerate(vocab)}
#
#     @classmethod
#     def encode_board(cls, board_state):
#         # BoardState = namedtuple('BoardState', 'board active_player castles')
#         board_string = ''
#         for c in board_state.board:
#             if c.isdigit() in list(range(1,9)):
#                 board_string += '.' * int(c)
#             else:
#                 board_string += c
#
#         print(board_state)
#         print(board_string)
#         assert False
#
#
#         input = input.split(' ')
#
#         input[1] = cls.vocab_to_tokens[input[1]]
#         input[2] = cls.vocab_to_tokens[input[2]]
#
#         for i in range(8):
#             input[0] = input[0].replace(f'{i+1}', cls.dots[i])
#
#         encoded_input = list(input[0])
#
#         for i in range(len(encoded_input)):
#             if encoded_input[i] in cls.vocab_to_num:
#                 encoded_input[i] = cls.vocab_to_num[encoded_input[i]]
#             else:
#                 encoded_input[i] = 0
#
#         encoded_input.append(input[1])
#         encoded_input.append(input[2])
#         return encoded_input
#
#     @classmethod
#     def decode_board(cls, input):
#         pass
#         # board = input[:-2]
#         # player = input[-2]
#         # player = cls.num_to_vocab[player]
#         # castling = input[-1]
#         # casting = cls.num_to_vocab[castling]
#         # for i in range(len(board)):
#         #     if board[i] in cls.num_to_vocab:
#         #         board[i] = cls.num_to_vocab[board[i]]
#         #     else:
#         #         board[i] = ' '
#         # board = ''.join(board)
#         # for i in range(8):
#         #     board = board.replace(cls.dots[7-i], f'{8-i}')
#         #
#         # return ' '.join([board, player, casting, '-', '0', '0'])
#
#
#
# def chess_tokenizer_test():
#     x = ChessTokenizer.encode('8/8/8/8/8/8/8/8 w KQkq - 0 0')
#     y = ChessTokenizer.decode(x)
#     assert x == y, "Error in tokenizer"
#
#
# # print(ChessTokenizer.vocab_to_tokens)
# # print(ChessTokenizer.tokes_to_vocab)
#
