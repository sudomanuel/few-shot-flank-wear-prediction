"""Locally adjusted exponent and extrapolation-weighted fit (sensitivity configurations).

The functions below are taken unchanged from the authors' wear-model code; only the file
layout and the data paths differ."""

import numpy as np
from .fewshot import theil_sen


def wls_gamma(tau, y, gamma):
    w = np.maximum(tau, 1e-9) ** gamma
    W = np.sqrt(w)
    A = np.column_stack([W, W * tau])
    c, *_ = np.linalg.lstsq(A, W * y, rcond=None)
    return float(c[1]), float(c[0])


def local_p_grid(o, v, m, p_star, fitfn, lam=200.0):
    best_p, best = p_star, np.inf
    for pc in np.arange(max(p_star - 0.15, 0.05), p_star + 0.1501, 0.05):
        tau = o[:m] ** pc
        a, b = theil_sen(tau, v[:m])
        sse = float(np.sum((b + a * tau - v[:m]) ** 2)) + lam * (pc - p_star) ** 2
        if sse < best:
            best, best_p = sse, pc
    return best_p
