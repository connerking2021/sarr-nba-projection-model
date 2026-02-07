"""
Add 'prob_over_25': estimated P(PTS > 25) using rolling 10-game mean and std (normal assumption).
"""
import pandas as pd
import numpy as np
from scipy.stats import norm


def add_prob_over_25(df: pd.DataFrame, pts_col: str = "PTS", window: int = 10, threshold: float = 25.0) -> pd.DataFrame:
    """
    Add column 'prob_over_25' = P(PTS > threshold) under normal approximation
    using rolling mean and rolling std over `window` games.
    """
    roll_mean = df[pts_col].rolling(window=window, min_periods=1).mean()
    roll_std = df[pts_col].rolling(window=window, min_periods=1).std()
    prob_over = np.where(
        roll_std > 0,
        1.0 - norm.cdf((threshold - roll_mean) / roll_std),
        np.where(
            roll_std == 0,
            np.where(roll_mean > threshold, 1.0, np.where(roll_mean < threshold, 0.0, 0.5)),
            np.nan,
        ),
    )
    df = df.copy()
    df["prob_over_25"] = prob_over
    return df
