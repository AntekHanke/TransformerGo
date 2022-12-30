import pandas as pd


def check_cllp_quality(file):
    df = pd.read_pickle(file)
    for row in df.itertuples():
        input = row[1]
        target = row[2]
        break


check_cllp_quality("/home/tomasz/Research/subgoal_chess_data/cllp_datasets/prep_cllp/part_5/data_5.pkl")