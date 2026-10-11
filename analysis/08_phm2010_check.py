"""External check on the public PHM 2010 milling data (Appendix A, Table 5): the pre-specified
configuration (least-squares fleet exponent, Theil-Sen level and rate from the first m inspections)
against the fleet-average curve and linear extrapolation, for 10 and 5 training cutters.

Input:  data/phm2010/phm2010_wear_percut.csv (public)
Output: out/results/phm2010_base.csv
"""
import os, sys
import numpy as np, pandas as pd
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from flankwear import phm2010 as E
df = pd.read_csv(E.DATA); print(df.cutter.unique(), df.groupby("cutter").size().to_dict())
GRID = np.arange(0.20, 1.001, 0.05)


def fit_p_ls(tr):
    best_p, best = 0.5, np.inf
    for p in GRID:
        tot = 0.0
        for o, v in tr:
            A = np.column_stack([np.ones(len(o)), o ** p]); c, *_ = np.linalg.lstsq(A, v, rcond=None)
            tot += float(np.sum((A @ c - v) ** 2))
        if tot < best: best, best_p = tot, p
    return best_p


def ev(curves, m, model):
    names = sorted(curves); per, P, Y, ps = {}, [], [], []
    for held in names:
        tr = [curves[c] for c in names if c != held]; o, v = curves[held]
        if len(o) <= m: continue
        fut = np.arange(m, len(o))
        if model == "physics_base":
            p = fit_p_ls(tr); ps.append(p); a, b = E.theil_sen(o[:m] ** p, v[:m]); pred = b + a * o[fut] ** p
        elif model == "linear_ls":
            c = np.polyfit(o[:m], v[:m], 1); pred = np.polyval(c, o[fut])
        else:
            return E.evaluate(curves, m, model)[1:] + (None,)
        per[held] = float(np.mean(np.abs(pred - v[fut]))); P += list(pred); Y += list(v[fut])
    P, Y = np.array(P), np.array(Y)
    return float(np.mean(list(per.values()))), float(1 - np.sum((Y - P) ** 2) / np.sum((Y - Y.mean()) ** 2)), ps


rows = []
for K in (10, 5):
    cur = E.subsample(df, K)
    for m in (3, 4):
        for model in ("physics_base", "physics", "avgcurve", "linear", "linear_ls"):
            mae, r2, ps = ev(cur, m, model)
            rows.append(dict(K=K, m=m, model=model, MAE=round(mae, 1), R2=round(r2, 2), p_star=ps))
R = pd.DataFrame(rows); os.makedirs(os.path.join(ROOT, "out", "results"), exist_ok=True); R.to_csv(os.path.join(ROOT, "out", "results", "phm2010_base.csv"), index=False); print(R.to_string(index=False))
