import chess


def chess960_to_standard(move, board):
    castlings = {"e8h8": "e8g8", "e8a8": "e8c8", "e1h1": "e1g1", "e1a1": "e1c1"}
    if move.uci() not in castlings:
        return move
    else:
        move_str = str(move)
        kings_position = move_str[:2]
        if str(board.piece_at(chess.parse_square(kings_position))) == "K":
            return chess.Move.from_uci(castlings[move_str])
        else:
            return move
