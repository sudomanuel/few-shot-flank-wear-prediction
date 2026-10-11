"""Robustness of the few-shot forecasts: model variants, rate clipping, nested conformal bands and
sensitivity to the 300 um analysis limit.

Input:  out/results/doe_records.csv
Output: out/results/ablation.csv, negative_rate.csv, conformal_nested.csv, cutoff_sensitivity.csv
        (Table 4, Figure 8)
"""
import os, sys
import numpy as np, pandas as pd
from scipy.stats import beta

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "out", "results"); os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, os.path.join(ROOT, "src"))
from flankwear import fewshot as MC, exponent as OC
from flankwear.prognostic_system import fit_population, fewshot_offset, predict_vb

D = pd.read_csv(os.path.join(OUT, "doe_records.csv"))
D.loc[D.workpiece.between(173, 183), "tool"] = "T17L"
ALL = (D[D.measured & (D.tool != "T17")][["tool", "part", "vb"]]
       .rename(columns={"tool": "tool_id", "part": "order"}).sort_values(["tool_id", "order"]).reset_index(drop=True))
d = ALL[ALL.vb <= 300].reset_index(drop=True)
LONG = {"T14", "T16", "T18", "T17L"}
tools_of = lambda dd: sorted(dd.tool_id.unique(), key=lambda t: (len(t), t))
GRID0 = np.arange(0.20, 1.001, 0.05)


def fit_p(tr, grid=GRID0, cens=300.0):
    best_p, best = 0.5, np.inf
    for p in grid:
        tot = 0.0
        for _, g in tr.groupby("tool_id"):
            gg = g[g.vb <= cens].sort_values("order")
            if len(gg) < 2: continue
            A = np.column_stack([np.ones(len(gg)), gg.order.to_numpy(float) ** p])
            c, *_ = np.linalg.lstsq(A, gg.vb.to_numpy(float), rcond=None)
            tot += float(np.sum((A @ c - gg.vb.to_numpy(float)) ** 2))
        if tot < best: best, best_p = tot, p
    return best_p


def ts(x, y): return MC.theil_sen(x, y)


def eb_prior(tr, p):
    """Per-tool OLS (a_k, b_k) in tau = t^p over training tools -> mean, covariance; pooled noise var."""
    co, rs = [], []
    for _, g in tr.groupby("tool_id"):
        if len(g) < 3: continue
        A = np.column_stack([np.ones(len(g)), g.order.to_numpy(float) ** p]); y = g.vb.to_numpy(float)
        c, *_ = np.linalg.lstsq(A, y, rcond=None); co.append(c); rs.append(y - A @ c)
    co = np.array(co); mu = co.mean(0); S = np.cov(co.T) + 1e-6 * np.eye(2)
    r = np.concatenate(rs); s2 = float(np.sum(r ** 2) / max(len(r) - 2 * len(co), 1))
    return mu, S, s2


def predict(dd, tt, m, model, o_eval=None, tr=None, cens=300.0):
    tr = dd[dd.tool_id != tt] if tr is None else tr
    g = dd[dd.tool_id == tt].sort_values("order"); o, v = g.order.to_numpy(float), g.vb.to_numpy(float)
    oe = o if o_eval is None else np.asarray(o_eval, float)
    kind, par = model
    if kind == "fleet":
        pop = fit_population(tr); return predict_vb(pop, oe, fewshot_offset(pop, o, v, m)), None
    if kind == "linear":
        c = np.polyfit(o[:m], v[:m], 1); return np.polyval(c, oe), None
    if kind == "persist":
        return np.full(len(oe), v[m - 1]), None
    if kind == "log":
        a, b = ts(np.log(o[:m]), v[:m]); return b + a * np.log(oe), a
    if kind == "fixed_p":
        p = par; a, b = ts(o[:m] ** p, v[:m]); return b + a * oe ** p, a
    if kind == "phys":
        p = fit_p(tr, GRID0 if par is None else par, cens); a, b = ts(o[:m] ** p, v[:m])
        return b + a * oe ** p, a
    if kind == "phys_pos":   # a-hat clipped at 0 (monotone enforced)
        p = fit_p(tr, GRID0, cens); a, b = ts(o[:m] ** p, v[:m])
        if a < 0: a, b = 0.0, float(np.median(v[:m]))
        return b + a * oe ** p, a
    if kind == "eb":         # random-coefficient power law, Gaussian prior from training tools
        p = fit_p(tr, GRID0, cens); mu, S, s2 = eb_prior(tr, p)
        A = np.column_stack([np.ones(m), o[:m] ** p]); Si = np.linalg.inv(S)
        post = np.linalg.solve(Si + A.T @ A / s2, Si @ mu + A.T @ v[:m] / s2)
        return post[0] + post[1] * oe ** p, post[1]
    raise ValueError(kind)


def evaluate(dd, m, model, cens=300.0):
    out = {}
    for tt in tools_of(dd):
        g = dd[dd.tool_id == tt].sort_values("order"); o, v = g.order.to_numpy(float), g.vb.to_numpy(float)
        if len(o) <= m: continue
        fut = np.arange(m, len(o)); fut = fut[v[fut] <= cens]
        if len(fut) == 0: continue
        pr, a = predict(dd, tt, m, model, o[fut], cens=cens)
        out[tt] = dict(err=pr - v[fut], true=v[fut], pred=pr, a=a)
    return out


def summarise(r):
    mae = np.mean([np.abs(x["err"]).mean() for x in r.values()])
    Y = np.concatenate([x["true"] for x in r.values()]); P = np.concatenate([x["pred"] for x in r.values()])
    lng = [np.abs(x["err"]).mean() for t, x in r.items() if t in LONG]
    return dict(n_tools=len(r), MAE=round(mae, 1), MAE_long=round(np.mean(lng), 1) if lng else np.nan,
                R2=round(1 - np.sum((Y - P) ** 2) / np.sum((Y - Y.mean()) ** 2), 2))


# ---------------- R1 + R2 + R5
MODELS = {"fleet-average curve": ("fleet", None), "linear extrapolation": ("linear", None),
          "persistence (last value)": ("persist", None),
          "physics-informed, fleet p* (grid 0.20-1.00)": ("phys", None),
          "physics-informed, fleet p* (grid 0.05-1.00)": ("phys", np.arange(0.05, 1.001, 0.05)),
          "physics-informed, a-hat clipped >= 0": ("phys_pos", None),
          "own-tool law, fixed p = 0.20": ("fixed_p", 0.20), "own-tool law, fixed p = 0.35": ("fixed_p", 0.35),
          "own-tool law, fixed p = 0.50": ("fixed_p", 0.50), "own-tool law, fixed p = 0.75": ("fixed_p", 0.75),
          "own-tool law, logarithmic": ("log", None),
          "random-coefficient power law (empirical Bayes)": ("eb", None)}
rows, neg = [], []
for m in (2, 3, 4):
    for name, mod in MODELS.items():
        r = evaluate(d, m, mod); rows.append(dict(m=m, model=name, **summarise(r)))
        if name.startswith("physics-informed, fleet p* (grid 0.20"):
            for t, x in r.items():
                if x["a"] is not None and x["a"] < 0: neg.append(dict(m=m, tool=t, a_hat=round(x["a"], 2), MAE=round(np.abs(x["err"]).mean(), 1)))
R = pd.DataFrame(rows); R.to_csv(os.path.join(OUT, "ablation.csv"), index=False)
N = pd.DataFrame(neg); N.to_csv(os.path.join(OUT, "negative_rate.csv"), index=False)
pd.set_option("display.width", 220); print(R.to_string(index=False)); print("\nnegative a-hat:\n", N.to_string(index=False))
ps05 = [fit_p(d[d.tool_id != t], np.arange(0.05, 1.001, 0.05)) for t in tools_of(d)]
print("extended grid p* per fold:", sorted(np.round(ps05, 2)))

# ---------------- R3 strictly nested conformal (m = 2, 3), Mondrian by horizon bin
def binof(h): return 0 if h <= 1 else (1 if h <= 3 else 2)
BINS = ["1 step", "2-3 steps", ">= 4 steps"]


def res_for(dd, tt, m, excl):
    g = dd[dd.tool_id == tt].sort_values("order"); v = g.vb.to_numpy(float)
    if len(v) <= m: return None
    tr = dd[~dd.tool_id.isin(excl | {tt})]
    pr, _ = predict(dd, tt, m, ("phys", None), tr=tr)
    return np.abs(pr[m:] - v[m:]), np.arange(1, len(v) - m + 1, dtype=float)


crow = []
for m in (2, 3):
    for nested in (False, True):
        tl = [t for t in tools_of(d) if len(d[d.tool_id == t]) > m]
        own = {t: res_for(d, t, m, set()) for t in tl}
        st = {b: dict(cov=[], w=[], inf=0, n=0) for b in range(3)}
        for t in tl:
            cal = []
            for u in tl:
                if u == t: continue
                ru = res_for(d, u, m, {t}) if nested else own[u]
                cal += list(zip(*ru))
            for rr, hh in zip(*own[t]):
                b = binof(hh); c = np.sort([x for x, y in cal if binof(y) == b]); n = len(c)
                k = int(np.ceil((n + 1) * 0.9)); q = np.inf if k > n else c[k - 1]
                s = st[b]; s["n"] += 1; s["cov"].append(rr <= q)
                if np.isfinite(q): s["w"].append(2 * q)
                else: s["inf"] += 1
        for b in range(3):
            s = st[b]
            if s["n"] == 0: continue
            k = int(np.sum(s["cov"])); n = s["n"]
            lo = beta.ppf(0.025, k, n - k + 1) if k > 0 else 0.0; hi = beta.ppf(0.975, k + 1, n - k) if k < n else 1.0
            crow.append(dict(m=m, calibration="nested (p* refitted without t and u)" if nested else "standard (p* of fold u)",
                             bin=BINS[b], points=n, covered=k, coverage=round(k / n, 3), ci=f"{lo:.2f}-{hi:.2f}",
                             unbounded=s["inf"], mean_width=round(np.mean(s["w"]), 1) if s["w"] else None))
C = pd.DataFrame(crow); C.to_csv(os.path.join(OUT, "conformal_nested.csv"), index=False); print("\n", C.to_string(index=False))

# ---------------- R4 sensitivity: 317 and 587 um readings included
srow = []
for m in (2, 3, 4):
    for name in ("fleet-average curve", "linear extrapolation", "physics-informed, fleet p* (grid 0.20-1.00)"):
        r = evaluate(ALL, m, MODELS[name], cens=1e9); srow.append(dict(m=m, model=name, data="all 104 readings", **summarise(r)))
S = pd.DataFrame(srow); S.to_csv(os.path.join(OUT, "cutoff_sensitivity.csv"), index=False); print("\n", S.to_string(index=False))
