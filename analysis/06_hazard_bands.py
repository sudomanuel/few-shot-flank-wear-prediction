"""Bootstrap bands of the breakage-hazard curve (resampling tools), for both codings of the 587 um
reading of T16.

Input:  out/results/doe_records.csv
Output: out/results/hazard_bands.csv (Figure 9)
"""
import os
import numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.stats import chi2

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "out", "results")
D = pd.read_csv(os.path.join(OUT, "doe_records.csv"))
D.loc[D.workpiece.between(173, 183), "tool"] = "T17L"     # the real 55/0.08/dry long run
D = D[D.tool != "T17"]                                     # short 55/0.08/dry run: no VBmax
H_MAX = 0.10
GRID = np.linspace(50, 650, 12001)


def risk_rows(scenario):
    out = []
    for t, g in D.groupby("tool", sort=False):
        g = g.sort_values("part")
        last = np.nan
        for _, r in g.iterrows():
            if t == "T16" and scenario == "S2" and r.part == 5:     # breakage recognised at part 5
                out.append(dict(tool=t, part=r.part, vb=last, y=1)); break
            ev = int(bool(r.breakage_row))
            out.append(dict(tool=t, part=r.part, vb=last if ev or np.isnan(r.vb) else r.vb, y=ev))
            if not np.isnan(r.vb) and not ev:
                last = r.vb
                out[-1]["vb"] = r.vb
            if ev:
                break
    R = pd.DataFrame(out).dropna(subset=["vb"])
    return R


def fit(R, firth=False, fix_b1=None):
    x = R.vb.to_numpy(float) / 100.0; y = R.y.to_numpy(float)
    A = np.column_stack([np.ones_like(x), x])

    def nll(b):
        if fix_b1 is not None:
            b = np.array([b[0], fix_b1])
        p = 1 / (1 + np.exp(-(A @ b))); p = np.clip(p, 1e-12, 1 - 1e-12)
        ll = np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
        if firth:
            W = p * (1 - p); ll += 0.5 * np.log(np.linalg.det(A.T @ (A * W[:, None])))
        return -ll
    x0 = [-4.0] if fix_b1 is not None else [-4.0, 0.5]
    r = minimize(nll, x0, method="Nelder-Mead", options=dict(xatol=1e-9, fatol=1e-11, maxiter=40000))
    b = np.array([r.x[0], fix_b1]) if fix_b1 is not None else r.x
    return b, r.fun


def h(b, vb):
    return 1 / (1 + np.exp(-(b[0] + b[1] * np.asarray(vb) / 100)))


def vbsafe(b):
    hh = h(b, GRID)
    if b[1] <= 0:
        return "none (risk does not rise with wear)" if hh[0] <= H_MAX else "none"
    below = GRID[hh <= H_MAX]
    return round(float(below.max()), 1) if len(below) else "none"


def profile_ci(R, b_hat, f_hat):
    crit = chi2.ppf(0.95, 1) / 2; lo = hi = np.nan
    for s in np.arange(b_hat[1], b_hat[1] - 8, -0.01):
        if fit(R, fix_b1=s)[1] - f_hat > crit: lo = s; break
    for s in np.arange(b_hat[1], b_hat[1] + 8, 0.01):
        if fit(R, fix_b1=s)[1] - f_hat > crit: hi = s; break
    return lo, hi



GRIDV = np.linspace(50, 400, 141)
rows = []
rng = np.random.default_rng(7)
for sc in ("S1", "S2"):
    R = risk_rows(sc); b, _ = fit(R)
    cols = {"scenario": sc, "vb": GRIDV, "h": h(b, GRIDV)}
    for t in R[R.y == 1].tool:
        bl, _ = fit(R[R.tool != t]); cols[f"loo_{t}"] = h(bl, GRIDV)
    tools = R.tool.unique(); curves = []
    for _ in range(500):
        pick = rng.choice(tools, len(tools), replace=True)
        Rb = pd.concat([R[R.tool == t] for t in pick])
        if Rb.y.sum() == 0: continue
        bb, _ = fit(Rb); curves.append(h(bb, GRIDV))
    curves = np.array(curves)
    cols["lo"] = np.percentile(curves, 2.5, axis=0); cols["hi"] = np.percentile(curves, 97.5, axis=0)
    rows.append(pd.DataFrame(cols))
pd.concat(rows).to_csv(os.path.join(OUT, "hazard_bands.csv"), index=False)
print("wrote hazard_bands.csv")
