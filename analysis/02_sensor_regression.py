"""Sensor-based regression on the measured VBmax labels under leave-one-tool-out validation.

Rows of the vibration-feature table whose workpiece has a measured VBmax in the DOE workbook are used,
excluding breakage parts, readings above 300 um and the unmeasured 55/0.08/dry short test
(workpieces 89-93). Seven regressors on all features and three PLS models on wear indicators.

Input:  data/vibration_features.csv (available from the authors on request), out/results/doe_records.csv
Output: out/results/sensor_regression.csv (Figure 5)
"""
import os, sys
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, ElasticNet, RidgeCV
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "out", "results"); os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, os.path.join(ROOT, "src"))
from flankwear import sensor_baseline as F2
_nc=F2.num_cols
F2.num_cols=lambda f: [c for c in _nc(f) if pd.api.types.is_numeric_dtype(f[c])]

pub = F2.load()
doe = pd.read_csv(os.path.join(OUT, "doe_records.csv"))
ok = doe[doe.measured & ~doe.breakage_row & (doe.vb <= 300)][["workpiece", "vb"]]
meas = pub.merge(ok, left_on="experiment_id", right_on="workpiece", how="inner")
meas = meas[meas.tool_id != "T17"].copy()
meas["vb_um"] = meas["vb"]; meas = meas.drop(columns=["workpiece", "vb"]).sort_values(["tool_id", "within_tool_order"]).reset_index(drop=True)
print(f"feature table {len(pub)} rows / {pub.tool_id.nunique()} tools; measured frame {len(meas)} rows / {meas.tool_id.nunique()} tools")


def mk(model):
    def f(Xtr, ytr, Xte, tr):
        return model().fit(Xtr, ytr).predict(Xte)
    return f


ZOO = {"Ridge (alpha=10)": F2.m_ridge, "Random forest (200 trees)": F2.m_rf,
       "LASSO (alpha=1)": mk(lambda: Lasso(alpha=1.0, max_iter=20000)),
       "Elastic net (alpha=1, l1=0.5)": mk(lambda: ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=20000)),
       "XGBoost (500 trees, lr=0.05)": mk(lambda: XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=4, colsample_bytree=0.9,
                                                                 subsample=0.9, random_state=0, verbosity=0)),
       "Ridge (CV alpha)": mk(lambda: RidgeCV(alphas=[0.1, 1, 10, 100, 1000])),
       "MLP (64-32)": mk(lambda: MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=2000, random_state=0, early_stopping=True))}
rows = []
for tag, f in (("measured labels only", meas),):
    allc = F2.num_cols(f); phys = F2.phys_cols(f)
    for name, fn in ZOO.items():
        r = F2.loto_predict(f, allc, fn)
        rows.append(dict(data=tag, features=f"all {len(allc)}", model=name, n_tools=r["n"], MAE=round(r["MAE"], 1), R2=round(r["R2"], 2)))
    for name, kw in (("PLS, physics indicators (absolute)", {}), ("PLS, physics indicators (relative to break-in)", dict(relative=True)),
                     ("PLS, relative + cutting conditions", dict(relative=True, add_cond=True))):
        r = F2.loto_predict(f, phys, F2.m_pls_factory(), **kw)
        rows.append(dict(data=tag, features=f"{len(phys)} physics indicators", model=name, n_tools=r["n"], MAE=round(r["MAE"], 1), R2=round(r["R2"], 2)))
    mono, trend, prog = F2.hi_quality(f, phys)
    rows.append(dict(data=tag, features="HI quality gate", model=f"monotonicity {mono:.2f} | trendability {trend:.2f} | prognosability {prog:.2f}"))
    print(tag, "done")
R = pd.DataFrame(rows); R.to_csv(os.path.join(OUT, "sensor_regression.csv"), index=False)
pd.set_option("display.width", 220); print(R.to_string(index=False))
