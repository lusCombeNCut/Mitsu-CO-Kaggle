"""This notebook code illustrates how to calculate the targets for the MITSUI&CO. Commodity Prediction Challenge from the (lagged) train or test features.
"""
import warnings

import numpy as np
import pandas as pd


def generate_log_returns(data, lag):
    log_returns = pd.Series(np.nan, index=data.index)

    # Compute log returns based on the rules
    for t in range(len(data)):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            try:
                log_returns.iloc[t] = np.log(data.iloc[t + lag + 1] / data.iloc[t + 1])
            except Exception:
                log_returns.iloc[t] = np.nan
    return log_returns


def generate_targets(column_a: pd.Series, column_b: pd.Series, lag: int) -> pd.Series:
    a_returns = generate_log_returns(column_a, lag)
    b_returns = generate_log_returns(column_b, lag)
    return a_returns - b_returns

"""
For example, to calculate target ID 423 for the train set you would call:

generate_targets(train['LME_CA_Close'], train['US_Stock_CCJ_adj_close'], 4)
"""