import chess
import numpy as np

from utils.data_utils import immutable_boards_to_img
from data_structures.data_structures import ImmutableBoard
from mcts.node_expansion import ChessStateExpander
from policy.chess_policy import LCZeroPolicy
from policy.cllp import CLLP
from subgoal_generator.subgoal_generator import BasicChessSubgoalGenerator
from utils.sample_fens import BLACK_MATE_IN_2
from value.chess_value import LCZeroValue

import sys

sys.path.append("/home/tomasz/Research/subgoal_search_chess")


def print_summary(input_immutable_board, generation_kwargs, k, raw_subgoals):
    print("=============================================================")
    print("Generation kwargs:", generation_kwargs)

    print(f"Input board: {input_immutable_board.fen}")

    lczero_policy = LCZeroPolicy()
    board = input_immutable_board.to_board()
    policy_moves = []
    for policy_move_num in range(k):
        policy_moves_all = lczero_policy.get_best_moves(board, 1)
        if len(policy_moves_all) == 0:
            break
        policy_move = policy_moves_all[0]
        policy_moves.append(policy_move)
        board.push(policy_move)

    policy_subgoal_stats = lczero_policy.get_path_probabilities(input_immutable_board, policy_moves)
    print(f"Policy subgoal stats: {policy_subgoal_stats} total = {np.prod(policy_subgoal_stats)}")

    print(f"Policy moves: {[chess.Move.uci(policy_move) for policy_move in policy_moves]}")
    policy_subgoal = ImmutableBoard.from_board(board)

    subgoals_info = expander.expand_state(
        input_immutable_board=b,
        cllp_num_beams=32,
        cllp_num_return_sequences=2,
        generator_num_beams=32,
        generator_num_subgoals=6,
        subgoal_distance_k=3,
        sort_subgoals_by="policy",
        **generation_kwargs,
    )


    subgoals = list(subgoals_info.keys())

    fig = immutable_boards_to_img(
        [b] + [policy_subgoal] + subgoals,
        ["input", f"policy {[x.uci() for x in policy_moves]}"] + [f"s{i}" for i in range(len(subgoals))],
    )
    fig.show()

    if raw_subgoals:
        return

    # fig = immutable_boards_to_img(
    #     [b] + [policy_subgoal] + subgoals,
    #     ["input", f"policy {[x.uci() for x in policy_moves]}"]
    #     + [f"s{i}:{[x.uci() for x in subgoals_info[subgoals[i]]['paths'][0]]}" for i in range(len(subgoals))],
    # )
    # fig.show()

    def show_subgoal_info(one_subgoal_info, num):
        print(
            f"Subgoal {num}: total_p = {one_subgoal_info['highest_total_probability']:.6f} "
            f"| min_p = {one_subgoal_info['highest_min_probability']:.6f} "
            f"| max_p = {one_subgoal_info['highest_max_probability']:.6f} "
            f"| value = {one_subgoal_info['value']:.6f}"
            f"| path = {[m.uci() for m in one_subgoal_info['paths'][0]]}"
            f"| path_p = {one_subgoal_info['path_raw_probabilities'][0]}"
        )

    print("*****************************************************")
    print(f"Summary correct subgoals: {len(subgoals_info)/generation_kwargs['num_return_sequences']}")
    # print(f"Summary: best total probability = {max([x['highest_total_probability'] for x in subgoals_info.values()])}")
    # print(f"Summary: best min probability = {max([x['highest_min_probability'] for x in subgoals_info.values()])}")

    for num, subgoal_info in enumerate(subgoals_info.values()):
        show_subgoal_info(subgoal_info, num)


MEDIUM_K3 = "/home/tomasz/Research/subgoal_chess_data/local_leela_models/4pgu_medium/k=3"
MEDIUM_LARGE_K3 = "/home/tomasz/Research/subgoal_chess_data/local_leela_models/generator_athena/k=3"

LONG_K_3 = "/home/tomasz/Research/subgoal_chess_data/local_leela_models/long_training/checkpoint-221500"
MEDIUM_LARGE_K1 = "/home/tomasz/Research/subgoal_chess_data/local_leela_models/generator_athena/k=1"

NEW_GEN_0 = "/home/tomasz/Research/subgoal_chess_data/models/generator_medium_v0"
NEW_GEN_1 = "/home/tomasz/Research/subgoal_chess_data/models/generator_medium_v1"
# NEW_GEN_2 = "/home/tomasz/Research/subgoal_chess_data/models/generator_medium_v2"
BABY_ENDINGS = "/home/tomasz/Research/subgoal_chess_data/models/baby_model_endings"


# EAGLE_OLD = "/home/tomasz/Research/subgoal_chess_data/local_leela_models/eagle_old_models"
EAGLE_SMALL = "/home/tomasz/Research/subgoal_chess_data/local_leela_models/eagle_old_models/k=3_small/final_model"
SMALL = "/home/tomasz/Research/subgoal_chess_data/local_leela_models/small_generator"

generator = BasicChessSubgoalGenerator(LONG_K_3)
cllp = CLLP("/home/tomasz/Research/subgoal_chess_data/local_leela_models/cllp/medium")
policy = LCZeroPolicy()
value = LCZeroValue()


expander = ChessStateExpander(policy, value, generator, cllp)

board = chess.Board(fen="r1b1kb1r/1pp4p/2p2p2/p3p3/4P1n1/2N2N2/PPP2PPP/R2Q2RK w Qkq - 0 31")


# board = chess.Board()
# board.push(chess.Move.from_uci("e2e4"))
# board.push(chess.Move.from_uci("e7e5"))
# board.push(chess.Move.from_uci("g1f3"))
# board.push(chess.Move.from_uci("b8c6"))
# board.push(chess.Move.from_uci("f1c4"))
# board.push(chess.Move.from_uci("g8f6"))
# board.push(chess.Move.from_uci("d2d4"))

b = ImmutableBoard.from_board(board)

# b = ImmutableBoard.from_fen_str(BLACK_MATE_IN_2)

# fig = immutable_boards_to_img([b], ["input"])
# fig.show()

# generation_kwargs = {"top_p": 0.999, "do_sample": True, "num_return_sequences": 4}
generation_kwargs = {}


print_summary(b, generation_kwargs, 8, False)
