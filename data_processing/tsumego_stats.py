import itertools
import os
import pickle
from typing import List

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.contingency_tables import mcnemar

from data_processing.goPlay.goban import (
    playingGoModel,
    TransformerPolicy,
    ConvolutionPolicy,
)
from data_processing.prepare_tsumego import tsumego_df

#example of definition of models
transformer_192k = TransformerPolicy(
    "/home/malgorzatarog/ProjectData/subgoal_chess_data/go_models/checkpoint-192500"
)
convolution_351k = ConvolutionPolicy(
    "/home/malgorzatarog/ProjectData/subgoal_chess_data/go_models/checkpoint-351500",
    history=True,
)


def get_tsumego_results(
    model: playingGoModel,
    tsumego: pd.DataFrame,
    out_path: str,
    filename: str,
) -> pd.DataFrame:
    results_list = []
    for _, puzzle in tsumego.iterrows():
        moves, probs = model.play_moves(puzzle["Game"])
        moves_probs = list(zip(moves, probs))
        moves_probs.sort(key=lambda x: x[1], reverse=True)
        top_1_move = np.NaN
        if puzzle["Coordinates_good"] == []:
            top_1_move = 0 if moves_probs[0][0] in puzzle["Coordinates_bad"] else 1
        elif puzzle["Coordinates_bad"] == []:
            top_1_move = 1 if moves_probs[0][0] in puzzle["Coordinates_good"] else 0
        else:
            for move, prob in moves_probs:
                if move in puzzle["Coordinates_good"]:
                    top_1_move = 1
                    break
                elif move in puzzle["Coordinates_bad"]:
                    top_1_move = 0
                    break
        prob_ok: float = 0
        prob_bad: float = 0
        prob_tenuki: float = 0
        for move, prob in zip(moves, probs):
            if puzzle["Coordinates_good"] == []:
                if move in puzzle["Coordinates_bad"]:
                    prob_bad += prob
                else:
                    prob_ok += prob
            elif puzzle["Coordinates_bad"] == []:
                if move in puzzle["Coordinates_good"]:
                    prob_ok += prob
                else:
                    prob_bad += prob
            else:
                if move in puzzle["Coordinates_good"]:
                    prob_ok += prob
                elif move in puzzle["Coordinates_bad"]:
                    prob_bad += prob
                else:
                    prob_tenuki += prob
        results_list.append(
            {
                "Type": puzzle["Type"],
                "Subtype": puzzle["Subtype"],
                "Role": puzzle["Role"],
                "Exact": puzzle["Exact"],
                "Locality": puzzle["Locality"],
                "Good": prob_ok,
                "Bad": prob_bad,
                "Tenuki": prob_tenuki,
                "Success_top_n": np.NaN
                if prob_ok + prob_bad == 0
                else prob_ok / (prob_ok + prob_bad),
                "Success_top_1": top_1_move,
            }
        )
    results_df = pd.DataFrame(results_list)
    pickle.dump(results_df, open(os.path.join(out_path, filename + ".pkl"), "wb"))
    return results_df


def get_tsumego_stats(
    results_1: pd.DataFrame,
    results_2: pd.DataFrame,
    categories: List[str],
    out_path: str,
    top_n: bool = True,
) -> pd.DataFrame:
    results = []
    category_lists = [results_1[category].unique().tolist() for category in categories]
    type_list = list(itertools.product(*category_lists))
    for types in type_list:
        data_1 = results_1
        data_2 = results_2
        for type, category in zip(types, categories):
            data_1 = data_1[data_1[category] == type]
            data_2 = data_2[data_2[category] == type]
        if top_n:
            data_1 = data_1["Success_top_n"]
            data_2 = data_2["Success_top_n"]
        else:
            data_1 = data_1["Success_top_1"]
            data_2 = data_2["Success_top_1"]
        data = pd.concat({"Model 1": data_1, "Model 2": data_2}, axis=1)
        data.dropna(inplace=True)
        mean_1 = data["Model 1"].mean()
        mean_2 = data["Model 2"].mean()
        if top_n:
            if np.all(data["Model 1"] == data["Model 2"]):
                p_value = np.NaN
            elif mean_1 > mean_2:
                p_value = wilcoxon(x=data["Model 1"], y=data["Model 2"], alternative="greater").pvalue
            else:
                p_value = wilcoxon(x=data["Model 2"], y=data["Model 1"], alternative="greater").pvalue
        else:
            if np.all(data["Model 1"] == data["Model 2"]):
                p_value = np.NaN
            else:
                crosstab = pd.crosstab(data["Model 1"], data["Model 2"])
                p_value = mcnemar(crosstab, exact=True).pvalue
        count = 0
        for a in [0.05, 0.01, 0.001]:
            if p_value < a:
                count += 1
        results.append(
            {
                "puzzle name": types,
                "number of samples": len(data),
                "success mean 1": f"{mean_1:.4f}",
                "success mean 2": f"{mean_2:.4f}",
                "p_value": f"{p_value:.4f}{'*' * count}",
            }
        )
    results_df = pd.DataFrame(results)
    filename = "results_"
    if top_n:
        filename += "top_n_"
    else:
        filename += "top_1_"
    filename += "_".join(categories)
    pickle.dump(results_df, open(os.path.join(out_path, filename + ".pkl"), "wb"))
    results_df.to_csv(os.path.join(out_path, filename +".csv"), index=False)
    return results_df


def tsumego_pipeline(
    model1: playingGoModel,
    model2: playingGoModel,
    tsumego: pd.DataFrame,
    categories: List[str],
    out_path: str,
    top_n: bool = True,
) -> pd.DataFrame:
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    try:
        results_1 = pickle.load(open(os.path.join(out_path, "results_1.pkl"), "rb"))
    except FileNotFoundError:
        results_1 = get_tsumego_results(model1, tsumego, out_path, "results_1")
    try:
        results_2 = pickle.load(open(os.path.join(out_path, "results_2.pkl"), "rb"))
    except FileNotFoundError:
        results_2 = get_tsumego_results(model2, tsumego, out_path, "results_2")

    results = get_tsumego_stats(results_1, results_2, categories, out_path, top_n)

    return results
