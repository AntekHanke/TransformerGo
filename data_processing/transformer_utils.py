

class ChessTokenizer:

    symbols = [' ', 'P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k', '/', '.']
    sym_to_num = {symbol: i for i, symbol in enumerate(symbols)}
    num_to_sym = {i: symbol for i, symbol in enumerate(symbols)}
    dots = ['.','..','...','....','.....','......','.......','........']



    @classmethod
    def encode(cls, input):

        input = input.split(' ')
        board = input[0]
        active_player = input[1]
        castling_avb = input[2]

        for i in range(7):
            board = board.replace(f'{8-i}', f'{7-i}.')
        board = board.replace('1', '.')

        encoded_board = list(board)
        for i in range(len(encoded_board)):
            if encoded_board[i] in cls.sym_to_num:
                encoded_board[i] = cls.sym_to_num[encoded_board[i]]
            else:
                encoded_board[i] = 0

        return encoded_board

    @classmethod
    def decode(cls, input):
        for i in range(len(input)):
            if input[i] in cls.num_to_sym:
                input[i] = cls.num_to_sym[input[i]]
            else:
                input[i] = ' '
        board = ''.join(input)
        for i in range(len(cls.dots)):
            board = board.replace(cls.dots[len(cls.dots)-1-i], f'{8-i}')
        return board

# string = "r1b1q1r1/p2nbk2/4pp1Q/1p1p3B/2pP3N/1PP1P3/P4PPP/R4RK1 b - -"
#
# print(string.replace('1', '.'))
# tokens = ChessTokenizer.encode(string)
# print(tokens)
# string = ChessTokenizer.decode(tokens)
# print(string)