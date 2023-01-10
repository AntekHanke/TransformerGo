import chess

from data_processing.data_utils import immutable_boards_to_img
from data_structures.data_structures import ImmutableBoard
from mcts.node_expansion import ChessStateExpander
from policy.chess_policy import LCZeroPolicy
from policy.cllp import CLLP
from subgoal_generator.subgoal_generator import BasicChessSubgoalGenerator
from value.chess_value import LCZeroValue


def print_summary(input_immutable_board, generation_kwargs):
    print(f"Input board: {input_immutable_board.fen}")

    subgoals_info = expander.expand_state(
        input_immutable_board=b, n_subgoals=1, cllp_num_beams=32, cllp_num_return_sequences=1, **generation_kwargs
    )

    subgoals = list(subgoals_info.keys())
    fig = immutable_boards_to_img(
        [b] + subgoals,
        ["input"] + [f"s{i}:{[x.uci() for x in subgoals_info[subgoals[i]]['paths'][0]]}" for i in range(len(subgoals))],
    )
    fig.show()

    def show_subgoal_info(one_subgoal_info, num):
        print(
            f"Subgoal {num}: total_p = {one_subgoal_info['highest_total_probability']:.6f} "
            f"| min_p = {one_subgoal_info['highest_min_probability']:.6f} "
            f"| max_p = {one_subgoal_info['highest_max_probability']:.6f} "
            f"| value = {one_subgoal_info['value']:.6f}"
            f"| path = {[m.uci() for m in one_subgoal_info['paths'][0]]}"
            f"| path_p = {one_subgoal_info['path_raw_probabilities'][0]}"
        )


    for num, subgoal_info in enumerate(subgoals_info.values()):
        show_subgoal_info(subgoal_info, num)



generator = BasicChessSubgoalGenerator(
    "/home/tomasz/Research/subgoal_chess_data/local_leela_models/generator_athena/k=3"
)
cllp = CLLP("/home/tomasz/Research/subgoal_chess_data/local_leela_models/cllp/medium")
policy = LCZeroPolicy()
value = LCZeroValue()


expander = ChessStateExpander(policy, value, generator, cllp)

# board = chess.Board()

board = chess.Board(fen="r1b1kb1r/1pp4p/2p2p2/p3p3/4P1n1/2N2N2/PPP2PPP/R2Q2RK w Qkq - 10 31")
# board.push(chess.Move.from_uci("e2e4"))
# board.push(chess.Move.from_uci("e7e5"))
# board.push(chess.Move.from_uci("g1f3"))
# board.push(chess.Move.from_uci("b8c6"))
# board.push(chess.Move.from_uci("f1c4"))
# board.push(chess.Move.from_uci("g8f6"))
# board.push(chess.Move.from_uci("d2d4"))

b = ImmutableBoard.from_board(board)
generation_kwargs = {"top_p": 0.9, "do_sample": True, "num_return_sequences": 4}


print_summary(b, generation_kwargs)
