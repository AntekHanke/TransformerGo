from configures.global_config import EAGLE_DATASET, EAGLE_HOME
from statistics.quality_of_subgoals import SubgoalQualityDatabaseGenerator
from subgoal_generator.subgoal_generator import BasicChessSubgoalGenerator


def generate_quality_database():
    k = 2
    quality_data_generator = SubgoalQualityDatabaseGenerator(
        k=k,
        n_subgoals=8,
        n_games=20,
        pgn_file="/home/tomek/Research/subgoal_chess_data/chess_data_aa",
        subgoal_generator=BasicChessSubgoalGenerator(
            f"/home/tomek/Research/subgoal_chess_data/eagle_model/k_2/checkpoint-11500"),
        take_transition_p=0.1,
        n_eval_datapoints=8,
        top_n_actions_max=3,
        check_exhaustive_search=True,
    )

    df = quality_data_generator.execute()
    df.to_csv(f"/home/tomek/Research/subgoal_chess_data/stats/subgoal_quality_database_k_{k}.csv", index=False)

generate_quality_database()