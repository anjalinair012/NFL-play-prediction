import numpy as np
import pandas as pd

KEYS = ["game_id", "play_id", "nfl_id", "frame_id"]


def rmse_yards_df(pred: pd.DataFrame, truth: pd.DataFrame) -> float:
    """RMSE in yards over x and y together.

    RMSE = sqrt( 1/(2N) * sum_i [ (x_i - x_hat_i)^2 + (y_i - y_hat_i)^2 ] )

    where N is the number of (game_id, play_id, nfl_id, frame_id) rows in `truth`.
    Both frames need the key columns plus x and y (use the output_*.csv frames as
    `truth`). Every truth row must have a prediction, as in the competition.
    """
    merged = truth[KEYS + ["x", "y"]].merge(
        pred[KEYS + ["x", "y"]], on=KEYS, how="left", suffixes=("", "_pred"), validate="one_to_one"
    )
    if merged[["x_pred", "y_pred"]].isna().any(axis=None):
        n = int(merged["x_pred"].isna().sum())
        raise ValueError(f"{n} truth rows have no (or NaN) prediction")

    dx = merged["x"].to_numpy(dtype=np.float64) - merged["x_pred"].to_numpy(dtype=np.float64)
    dy = merged["y"].to_numpy(dtype=np.float64) - merged["y_pred"].to_numpy(dtype=np.float64)
    return float(np.sqrt(np.mean(np.concatenate([dx, dy]) ** 2)))


