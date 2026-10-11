"""Few-shot wear prediction on the measured VBmax values under leave-one-tool-out validation.

Physics-informed model (fleet exponent, Theil-Sen level and rate from the first m inspections), the
fleet-average curve and linear extrapolation, for m = 2, 3 and 4; split-conformal bands.

Input:  out/results/doe_records.csv
Output: out/results/fewshot_doe_summary.csv, fewshot_doe_pertool.csv, fewshot_doe_conformal.csv
        (Table 3, Figures 6 and 7)
"""
import os, sys
import numpy as np, pandas as pd
from scipy.stats import binomtest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "out", "results")
sys.path.insert(0, os.path.join(ROOT, "src"))
from flankwear import fewshot as MC, exponent as OC, weighting as RA2
from flankwear.prognostic_system import fit_population, fewshot_offset, predict_vb

D = pd.read_csv(os.path.join(OUT, "doe_records.csv"))
D.loc[D.workpiece.between(173, 183), "tool"] = "T17L"
d = (D[D.measured & (D.tool != "T17") & (D.vb <= 300)][["tool", "part", "vb"]]
     .rename(columns={"tool": "tool_id", "part": "order"}).sort_values(["tool_id", "order"]).reset_index(drop=True))
LONG = {"T14", "T16", "T18", "T17L"}
CENSOR = 300.0
MC.tools_of = lambda dd: sorted(dd.tool_id.unique(), key=lambda t: (len(t), t))
OC.tools_of = MC.tools_of


def run(dd, m, model, **kw):
    out = {}
    for tt in MC.tools_of(dd):
        tr = dd[dd.tool_id != tt]; g = dd[dd.tool_id == tt].sort_values("order")
        o, v = g.order.to_numpy(float), g.vb.to_numpy(float)
        if len(o) <= m:
            continue
        fut = np.arange(m, len(o)); fut = fut[v[fut] <= CENSOR]
        if len(fut) == 0:
            continue
        if model == "baseline":
            pop = fit_population(tr); off = fewshot_offset(pop, o, v, m); pred = predict_vb(pop, o[fut], off)
        elif model == "linear":
            c = np.polyfit(o[:m], v[:m], 1); pred = np.polyval(c, o[fut])
        else:
            p = OC.fit_p(tr, m=(m if kw.get("pm") else None))
            if kw.get("loc"):
                p = RA2.local_p_grid(o, v, m, p, MC.theil_sen)
            tau = o[:m] ** p
            if kw.get("gamma") is not None:
                a, b = RA2.wls_gamma(tau, v[:m], kw["gamma"])
            else:
                a, _ = OC.FIT[kw.get("fit", "theil_sen")](tau, v[:m]); b = float(np.median(v[:m] - a * tau))
            pred = b + a * o[fut] ** p
        out[tt] = dict(err=pred - v[fut], true=v[fut], pred=pred, h=(fut - (m - 1)).astype(float))
    return out


MODELS = {"fleet-average curve": ("baseline", {}),
          "linear in t": ("linear", {}),
          "physics base (Theil-Sen)": ("phys", {}),
          "physics m=3 cfg (Siegel+local p)": ("phys", dict(fit="siegel", loc=True)),
          "physics m=4 cfg (WLS g=3, m-matched, local p)": ("phys", dict(gamma=3.0, pm=True, loc=True))}
summ, per = [], []
for m in (2, 3, 4):
    R = {name: run(d, m, mod, **kw) for name, (mod, kw) in MODELS.items()}
    for name, r in R.items():
        if not r:
            continue
        mae = np.mean([np.abs(x["err"]).mean() for x in r.values()])
        Y = np.concatenate([x["true"] for x in r.values()]); P = np.concatenate([x["pred"] for x in r.values()])
        r2 = 1 - np.sum((Y - P) ** 2) / np.sum((Y - Y.mean()) ** 2)
        short = [np.abs(x["err"]).mean() for t, x in r.items() if t not in LONG]
        lng = [np.abs(x["err"]).mean() for t, x in r.items() if t in LONG]
        summ.append(dict(m=m, model=name, n_tools=len(r), n_short=len(short), n_long=len(lng), n_points=len(Y),
                         MAE_tool_mean=round(mae, 2), MAE_short=round(np.mean(short), 2) if short else np.nan,
                         MAE_long=round(np.mean(lng), 2) if lng else np.nan, pooled_R2=round(r2, 2)))
        for t, x in r.items():
            per.append(dict(m=m, model=name, tool=t, n_future=len(x["err"]), MAE=round(np.abs(x["err"]).mean(), 2)))
    # paired sign test physics base vs fleet curve
    a = R["physics base (Theil-Sen)"]; b = R["fleet-average curve"]
    common = [t for t in a if t in b]
    wins = sum(np.abs(a[t]["err"]).mean() < np.abs(b[t]["err"]).mean() for t in common)
    summ.append(dict(m=m, model=f"sign test physics base vs fleet: {wins}/{len(common)} wins",
                     MAE_tool_mean=round(binomtest(wins, len(common)).pvalue, 4) if common else np.nan))
S = pd.DataFrame(summ); P_ = pd.DataFrame(per)
S.to_csv(os.path.join(OUT, "fewshot_doe_summary.csv"), index=False); P_.to_csv(os.path.join(OUT, "fewshot_doe_pertool.csv"), index=False)
pd.set_option("display.width", 250)
print(S.to_string(index=False))
print(P_[P_.model.isin(["physics base (Theil-Sen)", "fleet-average curve", "linear in t"])]
      .pivot_table(index=["m", "tool"], columns="model", values="MAE").to_string())

# exponent stability on the measured data
ps = [OC.fit_p(d[d.tool_id != t]) for t in MC.tools_of(d)]
print("\nfleet exponent p per LOTO fold (DOE data):", sorted(set(np.round(ps, 2))), "| at grid edge 0.20:", sum(abs(p - 0.2) < 1e-9 for p in ps), "of", len(ps))

# LOTO-honest split-conformal band, base model at m=2 (the only budget all tools reach)
r = run(d, 2, "phys"); rows = []
for alpha in (0.1, 0.2):
    cov, wid = [], []
    for t in r:
        cal = np.concatenate([np.abs(r[u]["err"]) for u in r if u != t]); n = len(cal)
        k = int(np.ceil((n + 1) * (1 - alpha)))
        q = np.inf if k > n else np.sort(cal)[k - 1]
        inside = np.abs(r[t]["err"]) <= q
        cov.append(inside.mean()); wid.append(2 * q)
    pts = np.concatenate([np.abs(r[t]["err"]) for t in r])
    rows.append(dict(m=2, nominal=1 - alpha, n_tools=len(r), n_residuals=len(pts),
                     coverage_tool_mean=round(np.mean(cov), 3), mean_full_width_um=round(np.mean(wid), 1)))
C = pd.DataFrame(rows); C.to_csv(os.path.join(OUT, "fewshot_doe_conformal.csv"), index=False)
print("\n", C.to_string(index=False))
