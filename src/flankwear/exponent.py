"""Fleet wear exponent p* (least-squares grid search over the training tools) and the
few-shot level/rate estimators (Theil-Sen, repeated median, least squares).

The functions below are taken unchanged from the authors' wear-model code; only the file
layout and the data paths differ."""

import numpy as np


CENSOR = 300.0


P_GRID = np.arange(0.20, 1.001, 0.05)


def tools_of(d):
    return sorted(d.tool_id.unique(), key=lambda t: int(str(t).lstrip("T") or 0))


def fit_p(tr, m=None):
    """Global exponent on training tools. If m is given, use only each training tool's first m
    points (m_matched strategy: consistent with what the model sees at deploy time)."""
    best_p, best = 0.5, np.inf
    for p in P_GRID:
        tot = 0.0
        for _, g in tr.groupby("tool_id"):
            gg = g[g.vb <= CENSOR].sort_values("order")
            if m is not None:
                gg = gg.iloc[:m]
            if len(gg) < 2:
                continue
            A = np.column_stack([np.ones(len(gg)), gg.order.to_numpy(float) ** p])
            c, *_ = np.linalg.lstsq(A, gg.vb.to_numpy(float), rcond=None)
            tot += float(np.sum((A @ c - gg.vb.to_numpy(float)) ** 2))
        if tot < best:
            best, best_p = tot, p
    return best_p


def theil_sen(x, y):
    s = np.median([(y[j] - y[i]) / (x[j] - x[i])
                   for i in range(len(x)) for j in range(i + 1, len(x)) if x[j] != x[i]])
    return float(s), float(np.median(y - s * x))


def siegel(x, y):
    """Siegel repeated-median: per-point median slope, then median of those (more robust to a single
    leverage point than Sen's pooled-pairwise median, which matters at m=3-4)."""
    n = len(x)
    med_i = []
    for i in range(n):
        sl = [(y[j] - y[i]) / (x[j] - x[i]) for j in range(n) if j != i and x[j] != x[i]]
        if sl:
            med_i.append(np.median(sl))
    s = float(np.median(med_i)) if med_i else 0.0
    return s, float(np.median(y - s * x))


def ols(x, y):
    A = np.column_stack([np.ones(len(x)), x])
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(c[1]), float(c[0])


FIT = {"theil_sen": theil_sen, "siegel": siegel, "ols": ols}
