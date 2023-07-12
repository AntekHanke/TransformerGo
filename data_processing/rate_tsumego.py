import pandas as pd

from data_structures.go_data_structures import GoImmutableBoard
from katago.katago_classes import KataGoBackend


def rate_tsumego(tsumego_df: pd.DataFrame) -> pd.DataFrame:
    results = []
    model = KataGoBackend()

    for _, puzzle in tsumego_df.iterrows():
        game = puzzle["Game"]
        move_distribution = model.policy_distribution_dict(immutable_board=GoImmutableBoard.from_game(game))
        player = game.get_active_player()
        good = 0
        bad = 0
        if not puzzle["Coordinates_good"]:
            for move in puzzle["Coordinates_bad"]:
                bad += move_distribution[(move[0], move[1], player)]
            good = 1-bad
        elif not puzzle["Coordinates_bad"]:
            for move in puzzle["Coordinates_good"]:
                good += move_distribution[(move[0], move[1], player)]
            bad = 1-good
        else:
            for move in puzzle["Coordinates_good"]:
                good += move_distribution[(move[0], move[1], player)]
            for move in puzzle["Coordinates_bad"]:
                bad += move_distribution[(move[0], move[1], player)]
        results.append(good/(bad+good))

    tsumego_df["KataGo"] = results
    return tsumego_df
