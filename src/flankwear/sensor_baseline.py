"""Vibration-feature regression under leave-one-tool-out validation (sensor benchmark).

The functions below are taken unchanged from the authors' wear-model code; only the file
layout and the data paths differ."""

import numpy as np
import os
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


CENSOR = 300.0


FEAT = os.path.join(ROOT, "data", "vibration_features.csv")


PHYS = ["rms", "energy", "kurtosis", "crest_factor", "dominant_freq"]


def load():
    f = pd.read_csv(FEAT)
    f = f.sort_values(["tool_id", "within_tool_order"]).reset_index(drop=True)
    return f


def tools_of(f):
    return sorted(f.tool_id.unique(), key=lambda t: int(str(t).lstrip("T") or 0))


def num_cols(f):
    return [c for c in f.columns if c not in ("tool_id", "within_tool_order", "vb_um", "experiment_id")
            and f[c].dtype != object]


def phys_cols(f):
    cols = [c for c in num_cols(f) if c.endswith("__mean")
            and any(p in c.lower() for p in PHYS) and (c.startswith("A_") or c.startswith("R_"))]
    return cols


def relative_to_breakin(f, cols):
    """x_tilde = x / x(first cut of the tool). Available at deploy (every tool has a break-in cut)."""
    g = f.copy()
    for tt, idx in f.groupby("tool_id").groups.items():
        ref = f.loc[idx].sort_values("within_tool_order").iloc[0]
        for c in cols:
            r = ref[c]
            g.loc[idx, c] = f.loc[idx, c] / r if abs(r) > 1e-9 else 1.0
    return g.replace([np.inf, -np.inf], np.nan).fillna(1.0)


def loto_predict(f, cols, model_fn, relative=False, add_cond=False):
    """Leakage-safe LOTO: train on other tools, predict held-out tool's per-cut VB from its features."""
    data = relative_to_breakin(f, cols) if relative else f.copy()
    if add_cond:
        cols = cols + ["vc", "fz"]
    P, Y, per_mae = [], [], []
    for tt in tools_of(f):
        tr = data[(data.tool_id != tt) & (data.vb_um <= CENSOR)]
        te = data[(data.tool_id == tt) & (data.vb_um <= CENSOR)]
        if len(te) == 0 or len(tr) < 8:
            continue
        Xtr = tr[cols].to_numpy(float); ytr = tr.vb_um.to_numpy(float)
        Xte = te[cols].to_numpy(float); yte = te.vb_um.to_numpy(float)
        sc = StandardScaler().fit(Xtr)
        pred = model_fn(sc.transform(Xtr), ytr, sc.transform(Xte), tr)
        P.append(pred); Y.append(yte); per_mae.append(np.abs(pred - yte).mean())
    P, Y = np.concatenate(P), np.concatenate(Y)
    r2 = 1 - np.sum((Y - P) ** 2) / np.sum((Y - Y.mean()) ** 2)
    return dict(MAE=float(np.mean(per_mae)), R2=float(r2), n=len(per_mae))


def m_ridge(Xtr, ytr, Xte, tr):
    return Ridge(alpha=10.0).fit(Xtr, ytr).predict(Xte)


def m_rf(Xtr, ytr, Xte, tr):
    return RandomForestRegressor(n_estimators=200, random_state=0).fit(Xtr, ytr).predict(Xte)


def m_pls_factory(kmax=4):
    def fit(Xtr, ytr, Xte, tr):
        # inner leave-one-tool-out to choose n_components
        best_k, best = 1, np.inf
        tools = tr.tool_id.to_numpy()
        for k in range(1, min(kmax, Xtr.shape[1]) + 1):
            errs = []
            for ut in np.unique(tools):
                itr, ite = tools != ut, tools == ut
                if ite.sum() == 0 or itr.sum() < 5:
                    continue
                pls = PLSRegression(n_components=k).fit(Xtr[itr], ytr[itr])
                errs.append(np.abs(pls.predict(Xtr[ite]).ravel() - ytr[ite]).mean())
            if errs and np.mean(errs) < best:
                best, best_k = np.mean(errs), k
        return PLSRegression(n_components=best_k).fit(Xtr, ytr).predict(Xte).ravel()
    return fit


def hi_quality(f, cols):
    """Coble-Hines HI metrics on the relative-feature 1st PLS score as a health indicator (per tool).
    Robust to degenerate tools (constant VB, zero-variance features)."""
    data = relative_to_breakin(f, cols)
    mono, trend, fails = [], [], []
    for tt in tools_of(f):
        g = data[(data.tool_id == tt) & (data.vb_um <= CENSOR)].sort_values("within_tool_order")
        y = g.vb_um.to_numpy(float)
        if len(g) < 3 or len(np.unique(y)) < 2:
            continue
        X = g[cols].to_numpy(float)
        keep = X.std(0) > 1e-9
        if keep.sum() < 1:
            continue
        Xs = np.nan_to_num((X[:, keep] - X[:, keep].mean(0)) / X[:, keep].std(0))
        try:
            hi = PLSRegression(n_components=1).fit(Xs, y).x_scores_[:, 0]
        except Exception:
            continue
        d = np.diff(hi)
        if len(d) == 0:
            continue
        mono.append(np.mean(d >= 0) if np.mean(d) >= 0 else np.mean(d <= 0))
        trend.append(abs(np.corrcoef(np.arange(len(hi)), hi)[0, 1]) if np.std(hi) > 1e-9 else 0.0)
        fails.append(hi[-1])
    if not fails:
        return np.nan, np.nan, np.nan
    prog = 1 - np.std(fails) / (np.mean(np.abs(fails)) + 1e-9)
    return float(np.mean(mono)), float(np.mean(trend)), float(max(prog, 0.0))
