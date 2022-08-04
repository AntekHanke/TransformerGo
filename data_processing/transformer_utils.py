
class ChessTokenizer:

    dots = ['.' * i for i in range(1, 9)]

    symbols = [' ', 'P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k', '/', '.']
    castlings = ['KQkq', 'KQk', 'KQ', 'K', 'Qkq', 'Qk', 'Q', 'kq', 'k', 'q', 'Kkq', 'Kq', 'Kk', 'Qq',
                 'KQq', '-']
    players = ['w', 'b']
    vocab_to_num = {symbol: i for i, symbol in enumerate(symbols + castlings + players)}
    num_to_vocab = {i: symbol for i, symbol in enumerate(symbols + castlings + players)}

    @classmethod
    def encode(cls, input):

        input = input.split(' ')

        input[1] = cls.vocab_to_num[input[1]]
        input[2] = cls.vocab_to_num[input[2]]

        for i in range(8):
            input[0] = input[0].replace(f'{i+1}', cls.dots[i])

        encoded_input = list(input[0])

        for i in range(len(encoded_input)):
            if encoded_input[i] in cls.vocab_to_num:
                encoded_input[i] = cls.vocab_to_num[encoded_input[i]]
            else:
                encoded_input[i] = 0

        encoded_input.append(input[1])
        encoded_input.append(input[2])
        return encoded_input

    @classmethod
    def decode(cls, input):
        board = input[:-2]
        player = input[-2]
        player = cls.num_to_vocab[player]
        castling = input[-1]
        casting = cls.num_to_vocab[castling]
        for i in range(len(board)):
            if board[i] in cls.num_to_vocab:
                board[i] = cls.num_to_vocab[board[i]]
            else:
                board[i] = ' '
        board = ''.join(board)
        for i in range(8):
            board = board.replace(cls.dots[7-i], f'{8-i}')

        return ' '.join([board, player, casting, '-', '-'])


def chess_tokenizer_test():
    x = ChessTokenizer.encode('8/8/8/8/8/8/8/8 w KQkq - -')
    y = ChessTokenizer.decode(x)
    assert x == y, "Error in tokenizer"


