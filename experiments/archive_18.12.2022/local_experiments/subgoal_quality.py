from statistics.quality_of_subgoals import SubgoalQualityDatabaseGenerator
from subgoal_generator.subgoal_generator import BasicChessSubgoalGenerator

k = 2
quality_data_generator = SubgoalQualityDatabaseGenerator(
    k=k,
    n_subgoals=6,
    n_games=500,
    pgn_file="/chess_data_aa",
    subgoal_generator=BasicChessSubgoalGenerator(
        "/generator_k_2/generator_model"),
    take_transition_p=0.1,
    n_eval_datapoints=75,
    check_exhaustive_search=True,
)

df = quality_data_generator.execute()
df.to_csv(f"/home/tomek/Research/subgoal_chess_data/subgoal_quality_database_k_{k}.csv", index=False)