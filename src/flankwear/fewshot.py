"""Robust Theil-Sen line fit and tool ordering used by the few-shot wear model.

The functions below are taken unchanged from the authors' wear-model code; only the file
layout and the data paths differ."""

import numpy as np


def theil_sen(x, y):
    s = np.median([(y[j] - y[i]) / (x[j] - x[i])
                   for i in range(len(x)) for j in range(i + 1, len(x)) if x[j] != x[i]])
    return float(s), float(np.median(y - s * x))


def tools_of(d):
    return sorted(d.tool_id.unique(), key=lambda t: int(str(t).lstrip("T") or 0))
