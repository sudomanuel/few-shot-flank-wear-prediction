"""Population fit of the wear law, few-shot adaptation of level and rate, and prediction.

The functions below are taken unchanged from the authors' wear-model code; only the file
layout and the data paths differ."""

import numpy as np


def fit_population(df_train):
    """Mean VB at each within-tool order over the TRAINING tools (the population wear curve).

    df_train: long-form DataFrame with columns [tool_id, order, vb].
    Returns {order:int -> mean vb} (the monotone-in-expectation population backbone).
    """
    g = df_train.groupby("order")["vb"].mean()
    return {int(o): float(v) for o, v in g.items()}


def _curve_at(pop, order):
    """Population VB at an integer order; nearest available order if missing (no extrapolation)."""
    if int(order) in pop:
        return pop[int(order)]
    keys = np.array(sorted(pop))
    return pop[int(keys[np.argmin(np.abs(keys - order))])]


def fewshot_offset(pop, orders_obs, vb_obs, m):
    """Per-tool offset from the first m early VB labels: mean residual to the population curve."""
    m = min(m, len(orders_obs))
    res = [vb_obs[i] - _curve_at(pop, orders_obs[i]) for i in range(m)]
    return float(np.mean(res))


def predict_vb(pop, orders, offset):
    """Population + few-shot offset prediction (the baseline-matching point estimate)."""
    return np.array([_curve_at(pop, o) + offset for o in orders], float)
