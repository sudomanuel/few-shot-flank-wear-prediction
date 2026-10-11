"""Dataset ledger from the DOE workbook (sheet DoE2): one row per machined part and one row per run.

Input:  data/DOE.xlsx (available from the authors on request) and data/tool_ids.csv
Output: out/results/doe_records.csv and out/results/doe_tools.csv (Table 2, Figure 3)
"""
import os
import numpy as np, pandas as pd, openpyxl

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
OUT = os.path.join(ROOT, "out", "results"); os.makedirs(OUT, exist_ok=True)

ws = openpyxl.load_workbook(os.path.join(ROOT, "data", "DOE.xlsx"), data_only=True)["DoE2"]
rows = []
for r in range(2, ws.max_row + 1):
    wp = ws.cell(r, 1).value
    if wp is None:
        continue
    vb = ws.cell(r, 11).value
    note = ws.cell(r, 20).value
    rows.append(dict(workpiece=int(wp), tool_no=ws.cell(r, 2).value, strategy=ws.cell(r, 3).value,
                     cooling=ws.cell(r, 5).value, vc=ws.cell(r, 6).value, f=ws.cell(r, 7).value,
                     vb_raw=vb, vb=float(vb) if isinstance(vb, (int, float)) else np.nan,
                     note=note))
D = pd.DataFrame(rows)

# condition-run id: consecutive workpieces sharing condition + strategy
key = D[["strategy", "cooling", "vc", "f"]].astype(str).agg("|".join, axis=1)
D["run"] = (key != key.shift()).cumsum()

# map runs to the analysis tool ids by experiment (= workpiece) number
man = pd.read_csv(os.path.join(ROOT, "data", "tool_ids.csv"))
wp2tool = dict(zip(man.experiment_id, man.tool_id))
D["analysis_tool"] = D.workpiece.map(wp2tool)
run2tool = D.dropna(subset=["analysis_tool"]).groupby("run").analysis_tool.first().to_dict()
D["tool"] = D.run.map(run2tool).fillna("(not analysed)")
D["part"] = D.groupby("run").cumcount() + 1
D["regime"] = np.where(D.strategy.str.startswith("Long"), "long", "short")
D["breakage_row"] = D.note.astype(str).str.contains("Breakage", case=False) | D.vb_raw.astype(str).eq("broken")
D["measured"] = D.vb.notna()
D.to_csv(os.path.join(OUT, "doe_records.csv"), index=False)

# ----------------------------------------------------------------- tool ledger
T = []
for run, g in D.groupby("run"):
    g = g.sort_values("part"); meas = g[g.measured]
    brk = g[g.breakage_row]
    is_brk = len(brk) > 0
    last_meas = meas.iloc[-1] if len(meas) else None
    T.append(dict(
        tool=g.tool.iloc[0], tool_no_col=sorted(g.tool_no.unique().tolist()),
        vc=g.vc.iloc[0], f=g.f.iloc[0], cooling=g.cooling.iloc[0], strategy=g.strategy.iloc[0],
        workpieces=f"{g.workpiece.min()}-{g.workpiece.max()}", n_parts=len(g), n_vb=int(g.measured.sum()),
        inspected_parts=",".join(str(p) for p in meas.part),
        vb_values=",".join(f"{v:g}" for v in meas.vb),
        first_vb=meas.vb.iloc[0] if len(meas) else np.nan,
        last_vb=last_meas.vb if last_meas is not None else np.nan,
        last_vb_part=int(last_meas.part) if last_meas is not None else np.nan,
        event="breakage" if is_brk else ("no VB recorded" if len(meas) == 0 else "right-censored (stopped after 5 parts)"),
        breakage_part=int(brk.part.iloc[0]) if is_brk else np.nan,
        failure_interval=(f"({int(last_meas.part)}, {int(brk.part.iloc[0])}]" if is_brk else ""),
        monotone=bool(np.all(np.diff(meas.vb.values) >= 0)) if len(meas) > 1 else True))
TL = pd.DataFrame(T)
TL.to_csv(os.path.join(OUT, "doe_tools.csv"), index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 80)
print(TL.to_string(index=False))
print("\nmeasured VBmax values:", int(D.measured.sum()), "| analysed runs:", (TL.tool != "(not analysed)").sum())
