from joblib import load
from data_processing.chess_data_generator import ChessSubgoalGamesDataGenerator, ELOFilter
from data_processing.chess_tokenizer import ChessTokenizer
from utils.data_utils import immutable_boards_to_img


data_generator = ChessSubgoalGamesDataGenerator(
    pgn_file="/home/gracjan/subgoal/subgoal_search_chess/assets/cas_small.pgn",
    chess_filter=ELOFilter,
    p_sample=1.0,
    max_games=1000,
    train_eval_split=0.5,
    log_stats_after_n_games=1,
    save_path_to_train_set="/home/gracjan/test_data_new_data_generator",
    save_path_to_eval_set="/home/gracjan/test_data_new_data_generator",
    save_data_every_n_games=10,
    range_of_k=[1, 2, 3, 4, 5, 6],
    number_of_datapoint_from_one_game=10,
)
data_generator.create_data()


# data = load("/home/gracjan/test_data_new_data_generator/subgoals_all_k/train/cas_small.pgn_train_part_0.pkl")
# t = ChessTokenizer()
# input_board_k_1 = t.decode_board(data.loc[0]["input_ids_1"])
# subgoal_k_1 = t.decode_board(data.loc[0]["labels_1"])
# input_board_k_2 = t.decode_board(data.loc[0]["input_ids_2"])
# subgoal_k_2 = t.decode_board(data.loc[0]["labels_2"])
# input_board_k_1_next = t.decode_board(data.loc[1]["input_ids_1"])
# subgoal_k_1_next = t.decode_board(data.loc[1]["labels_1"])
# input_board_k_2_next = t.decode_board(data.loc[1]["input_ids_2"])
# subgoal_k_2_next = t.decode_board(data.loc[1]["labels_2"])
#
# print(data.columns)
