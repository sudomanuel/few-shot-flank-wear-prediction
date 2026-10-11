"""Discrete-time breakage hazard on the measured data: 4 observed breakages, 14 right-censored short tests.

Risk set: one row per machined part; wear covariate = last VBmax measured at or before that part
(known at decision time, never a future value). The 55/0.08/dry short test (workpieces 89-93) has no
VBmax and cannot enter a wear hazard. Two codings of the 587 um reading of T16 are fitted:
  S1: valid VBmax, breakage at part 6 with covariate 587 um;
  S2: post-damage reading, breakage counted at part 5 with covariate 154 um (part 4).

Input:  out/results/doe_records.csv
Output: out/results/hazard_doe_fits.csv, hazard_doe_loo.csv, hazard_riskset_S1.csv, hazard_riskset_S2.csv
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


res, loo = [], []
for sc in ("S1", "S2"):
    R = risk_rows(sc)
    R.to_csv(os.path.join(OUT, f"hazard_riskset_{sc}.csv"), index=False)
    for firth in (False, True):
        b, f = fit(R, firth)
        lo, hi = profile_ci(R, b, f) if not firth else (np.nan, np.nan)
        res.append(dict(scenario=sc, estimator="Firth" if firth else "ML", rows=len(R), events=int(R.y.sum()),
                        tools=R.tool.nunique(), b0=round(b[0], 3), b1_per100um=round(b[1], 3),
                        b1_CI95=f"[{lo:.2f}, {hi:.2f}]" if not firth else "",
                        h_150=round(float(h(b, 150)), 4), h_200=round(float(h(b, 200)), 4),
                        h_250=round(float(h(b, 250)), 4), h_300=round(float(h(b, 300)), 4),
                        VBsafe_h010=vbsafe(b)))
    ev_tools = R[R.y == 1].tool.tolist()
    for drop in ev_tools:
        b, _ = fit(R[R.tool != drop])
        loo.append(dict(scenario=sc, dropped=drop, b1_per100um=round(b[1], 3), VBsafe_h010=vbsafe(b)))
    print(sc, "events:", R[R.y == 1][["tool", "part", "vb"]].to_dict("records"))
H = pd.DataFrame(res); L = pd.DataFrame(loo)
H.to_csv(os.path.join(OUT, "hazard_doe_fits.csv"), index=False); L.to_csv(os.path.join(OUT, "hazard_doe_loo.csv"), index=False)
pd.set_option("display.width", 250)
print(H.to_string(index=False)); print(L.to_string(index=False))

# tool-cluster bootstrap, scenario S2 (the more conservative coding of T16) and S1
rng = np.random.default_rng(2026)
for sc in ("S1", "S2"):
    R = risk_rows(sc); tools = R.tool.unique(); vals = []; nev = []; neg = 0
    for _ in range(1000):
        pick = rng.choice(tools, len(tools), replace=True)
        Rb = pd.concat([R[R.tool == t] for t in pick])
        k = int(Rb.y.sum()); nev.append(k)
        if k == 0:
            vals.append(np.nan); continue
        b, _ = fit(Rb)
        if b[1] <= 0:
            neg += 1; vals.append(np.inf); continue
        v = vbsafe(b); vals.append(v if isinstance(v, float) else np.nan)
    vals = np.array(vals, float); fin = np.isfinite(vals)
    print(f"\nbootstrap {sc}: zero-event resamples {np.mean(np.array(nev) == 0) * 100:.1f}% | "
          f"non-positive slope {neg / 10:.1f}% | finite VBsafe {fin.mean() * 100:.1f}% | "
          f"2.5/50/97.5 pct of finite: {np.round(np.percentile(vals[fin], [2.5, 50, 97.5]), 0)}")

# Kaplan-Meier on the wear scale (event = breakage at last pre-breakage VBmax; censored = last VBmax)
for sc, t16 in (("S1", 587.0), ("S2", 154.0)):
    ev = {"T14": 200.0, "T16": t16, "T18": 202.0, "T17L": 279.0}
    cens = D[(D.regime == "short") & D.measured].groupby("tool").vb.last().to_dict()
    pts = [(v, 1) for v in ev.values()] + [(v, 0) for v in cens.values()]
    S = 1.0; tab = []
    for w in sorted(set(v for v, e in pts if e == 1)):
        n = sum(1 for v, e in pts if v >= w); d = sum(1 for v, e in pts if v == w and e == 1)
        S *= 1 - d / n; tab.append(dict(scenario=sc, VB=w, at_risk=n, events=d, S=round(S, 3)))
    print(pd.DataFrame(tab).to_string(index=False))
    vals = np.array(sorted(ev.values()))
    print(f"{sc}: observed wear-at-breakage {vals} | range {vals.min():g}-{vals.max():g} | median {np.median(vals):g} | "
          f"Q10 type-7 {np.quantile(vals, .1):.1f} | Q10 inverse-ECDF {np.quantile(vals, .1, method='inverted_cdf'):g}")
