from configures.global_config import EAGLE_DATASET, EAGLE_HOME
from statistics.quality_of_subgoals import SubgoalQualityDatabaseGenerator
from subgoal_generator.subgoal_generator import BasicChessSubgoalGenerator


def generate_quality_database(k):
    quality_data_generator = SubgoalQualityDatabaseGenerator(
        k=k,
        n_subgoals=6,
        n_games=1500,
        pgn_file=EAGLE_DATASET,
        subgoal_generator=BasicChessSubgoalGenerator(
            f"{EAGLE_HOME}/chess_models/generator_k={k}-lr_0.0002/out/checkpoint-11500/"),
        take_transition_p=0.05,
        n_eval_datapoints=1000,
        check_exhaustive_search=True,
        stockfish_path="cluster",
        top_n_actions_max=3
    )

    df = quality_data_generator.execute()
    df.to_csv(f"{EAGLE_HOME}/stats/subgoal_quality_database_k_{k}_small.csv", index=False)
