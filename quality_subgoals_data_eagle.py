from configs.global_config import EAGLE_DATASET
from statistics.quality_of_subgoals import SubgoalQualityDatabaseGenerator
from subgoal_generator.subgoal_generator import BasicChessSubgoalGenerator


def generate_quality_database():

    k=2
    quality_data_generator = SubgoalQualityDatabaseGenerator(
        k=k,
        n_subgoals=6,
        n_games=500,
        pgn_file=EAGLE_DATASET,
        subgoal_generator=BasicChessSubgoalGenerator("/home/plgrid/plgtodrzygozdz/chess_models/generator_k=2-lr_0.0002/out/checkpoint-11500"),
        take_transition_p=0.05,
        n_eval_datapoints=50,
        check_exhaustive_search=True,
    )

    df = quality_data_generator.execute()
    df.to_csv(f"/home/plgrid/plgtodrzygozdz/subgoal_chess/subgoal_quality/quality_data_k={k}.csv", index=False)
