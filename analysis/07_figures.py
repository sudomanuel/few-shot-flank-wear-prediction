"""Figures 2-9 of the paper, computed from the result tables (no value is typed in by hand).
Figure 2 needs the micrographs in data/micrographs/ (available from the authors on request) and is
skipped otherwise. The values plotted are also written to out/figures/figure_data.json.

Input:  out/results/*.csv
Output: out/figures/Fig2_micrographs.png ... Fig9_breakage_evidence.png
"""
import os, sys, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap
from scipy.optimize import minimize

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
P2 = os.path.join(ROOT, "out", "results")
FIG = os.path.join(ROOT, "out", "figures"); os.makedirs(FIG, exist_ok=True)
sys.path.insert(0, os.path.join(ROOT, "src"))
from flankwear import fewshot as MC, exponent as OC, weighting as RA2
from flankwear.prognostic_system import fit_population, fewshot_offset, predict_vb

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#52514e",
                     "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e", "ytick.color": "#52514e",
                     "grid.color": "#e6e5e1", "grid.linewidth": 0.6, "legend.frameon": False, "savefig.dpi": 300})
C_PHYS, C_FLEET, C_LIN, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#8a8984"
SEQ = LinearSegmentedColormap.from_list("blue", ["#d6e6f8", "#2a78d6", "#0d2f5c"])
OUTD = {}

D = pd.read_csv(os.path.join(P2, "doe_records.csv"))
D.loc[D.workpiece.between(173, 183), "tool"] = "T17L"
LONG = ["T14", "T16", "T17L", "T18"]
d = (D[D.measured & (D.tool != "T17") & (D.vb <= 300)][["tool", "part", "vb"]]
     .rename(columns={"tool": "tool_id", "part": "order"}).sort_values(["tool_id", "order"]).reset_index(drop=True))
MC.tools_of = lambda dd: sorted(dd.tool_id.unique(), key=lambda t: (len(t), t)); OC.tools_of = MC.tools_of


def forecast(dd, tt, m, model, o_eval=None, excl=()):
    tr = dd[~dd.tool_id.isin(set(excl) | {tt})]; g = dd[dd.tool_id == tt].sort_values("order")
    o, v = g.order.to_numpy(float), g.vb.to_numpy(float)
    oe = o if o_eval is None else o_eval
    if model == "fleet":
        pop = fit_population(tr); off = fewshot_offset(pop, o, v, m); return predict_vb(pop, oe, off)
    if model == "linear":
        c = np.polyfit(o[:m], v[:m], 1); return np.polyval(c, oe)
    p = OC.fit_p(tr); a, b = MC.theil_sen(o[:m] ** p, v[:m]); return b + a * np.asarray(oe, float) ** p


def residuals(dd, m, model="phys", excl=()):
    out = {}
    for tt in MC.tools_of(dd):
        if tt in excl: continue
        g = dd[dd.tool_id == tt].sort_values("order"); v = g.vb.to_numpy(float)
        if len(v) <= m: continue
        pr = forecast(dd, tt, m, model, excl=excl)
        out[tt] = (np.abs(pr[m:] - v[m:]), np.arange(1, len(v) - m + 1, dtype=float))
    return out


def binof(h): return 0 if h <= 1 else (1 if h <= 3 else 2)
BINS = ["1 step", "2-3 steps", ">= 4 steps"]


def conformal(res, alpha=0.10, dd=None, m=None):
    """LOTO split conformal, Mondrian by horizon bin, rank ceil((n+1)(1-a)); infinite band if rank > n.
    Nested calibration: for held-out tool t, the residual of calibration tool u comes from a model fitted
    without t AND u (p* refitted), so the held-out tool influences nothing that is used to score it."""
    stats = {b: dict(cov=[], width=[], ncal=[], inf=0, npts=0) for b in range(3)}
    q_by_tool = {}
    for t, (r, h) in res.items():
        res_t = residuals(dd, m, excl=(t,)) if dd is not None else {u: x for u, x in res.items() if u != t}
        cal = [(rr, hh) for u, (R_, H_) in res_t.items() for rr, hh in zip(R_, H_)]
        qt = []
        for rr, hh in zip(r, h):
            b = binof(hh); c = np.sort([x for x, y in cal if binof(y) == b]); n = len(c)
            k = int(np.ceil((n + 1) * (1 - alpha))); q = np.inf if k > n else c[k - 1]
            s = stats[b]; s["npts"] += 1; s["ncal"].append(n); s["cov"].append(rr <= q)
            if np.isfinite(q): s["width"].append(2 * q)
            else: s["inf"] += 1
            qt.append(q)
        q_by_tool[t] = np.array(qt)
    summ = []
    for b in range(3):
        s = stats[b]
        if s["npts"] == 0: continue
        k = int(np.sum(s["cov"])); n = s["npts"]
        from scipy.stats import beta
        lo = beta.ppf(0.025, k, n - k + 1) if k > 0 else 0.0; hi = beta.ppf(0.975, k + 1, n - k) if k < n else 1.0
        summ.append(dict(bin=BINS[b], points=n, coverage=k / n, ci=[float(lo), float(hi)],
                         cal_n_median=float(np.median(s["ncal"])), cal_n_min=int(np.min(s["ncal"])),
                         mean_width=float(np.mean(s["width"])) if s["width"] else None, infinite_bands=s["inf"]))
    return summ, q_by_tool


# ------------------------------------------------------------------ Fig. 3 dataset overview
def fig3():
    T = pd.read_csv(os.path.join(P2, "doe_tools.csv"))
    T.loc[T.workpieces == "173-183", "tool"] = "T17L"; T.loc[T.workpieces == "89-93", "tool"] = "T17 (short)"
    T = T.sort_values(["vc", "f", "cooling", "strategy"], ascending=[False, False, False, True]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    xmap = lambda p: p if p <= 12 else 12 + (p - 12) * (8 / 67)   # compress parts 13-79
    for i, r in T.iterrows():
        y = len(T) - 1 - i
        g = D[D.workpiece.between(*map(int, r.workpieces.split("-")))].sort_values("workpiece")
        last = g.part.max()
        ax.plot([xmap(1), xmap(last)], [y, y], color="#d9d8d3", lw=4, solid_capstyle="round", zorder=1)
        m = g[g.measured]
        inreg = m[m.vb <= 300]; out = m[m.vb > 300]
        sc = ax.scatter([xmap(p) for p in inreg.part], [y] * len(inreg), c=inreg.vb, cmap=SEQ, vmin=60, vmax=300, s=22,
                        edgecolor="white", linewidth=0.6, zorder=3)
        ax.scatter([xmap(p) for p in out.part], [y] * len(out), facecolor="white", edgecolor=INK, s=22, linewidth=0.9, zorder=3)
        if r.event == "breakage":
            ax.scatter([xmap(r.breakage_part)], [y], marker="x", color="#e34948", s=34, linewidth=1.6, zorder=4)
        elif str(r.event).startswith("right"):
            ax.annotate("", xy=(xmap(last) + 0.9, y), xytext=(xmap(last) + 0.2, y),
                        arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.0))
        lab = f"{r.vc:g} / {r.f:g} / {r.cooling}" + ("  long" if r.strategy.startswith("Long") else "")
        ax.text(-0.3, y, f"{r.tool:>11}  {lab}", ha="right", va="center", fontsize=6.6, color=INK)
    ax.set_yticks([]); ax.spines["left"].set_visible(False)
    ticks = [1, 3, 5, 8, 12, 30, 50, 79]
    ax.set_xticks([xmap(t) for t in ticks]); ax.set_xticklabels([str(t) for t in ticks])
    ax.axvline(xmap(12.5), color="#e6e5e1", lw=0.8, ls=(0, (2, 2)))
    ax.set_xlim(0.3, xmap(80)); ax.set_xlabel("Part number (axis compressed beyond part 12)")
    cb = fig.colorbar(sc, ax=ax, fraction=0.025, pad=0.01); cb.set_label("VB$_{\\mathrm{max}}$ (µm)"); cb.outline.set_visible(False)
    h = [plt.Line2D([], [], marker="o", ls="", color=C_PHYS, label="VB$_{\\mathrm{max}}$ inspection (≤ 300 µm)"),
         plt.Line2D([], [], marker="o", ls="", mfc="white", mec=INK, label="inspection > 300 µm"),
         plt.Line2D([], [], marker="x", ls="", color="#e34948", label="tool breakage recorded"),
         plt.Line2D([], [], marker=">", ls="", color=MUTED, label="stopped after 5 parts (censored)")]
    ax.legend(handles=h, loc="upper center", bbox_to_anchor=(0.45, -0.13), ncol=2, fontsize=6.8)
    ax.set_title(r"$v_\mathrm{c}$ (m/min) / $f$ (mm/rev) / cooling", loc="left", fontsize=7, color=MUTED, x=-0.02)
    fig.subplots_adjust(left=0.30, right=0.93, bottom=0.24, top=0.95)
    fig.savefig(os.path.join(FIG, "Fig3_dataset_overview.png")); plt.close(fig)


# ------------------------------------------------------------------ Fig. 2 micrograph artwork
UM_PER_PX = None   # microscope calibration at the exported resolution (µm per pixel); scale bars are drawn only when known


def fig2():
    """Figure 2: three micrographs of the 55/0.08/dry long test (requires data/micrographs/)."""
    from PIL import Image
    src = os.path.join(ROOT, "data", "micrographs")
    if not os.path.isdir(src):
        print("Figure 2 skipped: micrographs not found in data/micrographs/"); return
    panels = [("micro_a_early_133um.jpg", "(a) Part 1", "VB$_{\\mathrm{max}}$ = 133 µm", "#1f5fa8"),
              ("micro_b_developed_220um.jpg", "(b) Part 5", "VB$_{\\mathrm{max}}$ = 220 µm", "#c87a00"),
              ("micro_c_chipping.jpg", "(c) Tool breakage", "recorded at part 11", "#b03030")]
    fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.05))
    for ax, (fn, head, sub, col) in zip(axes, panels):
        im = Image.open(os.path.join(src, fn)); w, h = im.size
        arr = np.asarray(im.convert("L")); dark = (arr < 28).mean(axis=1); ok = np.where(dark < 0.30)[0]
        y0, y1 = (int(ok[0]), int(ok[-1]) + 1) if len(ok) else (0, h)
        im = im.crop((0, y0, w, y1)); h = y1 - y0; sw = int(h * 4 / 3); x0 = 0 if sw >= w else max(0, int(w * 0.02))
        im = im.crop((x0, 0, min(w, x0 + sw), h - 6))
        ax.imshow(im); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_visible(False)
        ax.set_title(head, fontsize=8.5, fontweight="bold", color=col, pad=3)
        ax.text(0.5, -0.05, sub, transform=ax.transAxes, ha="center", va="top", fontsize=7.5, color=INK)
        if UM_PER_PX:   # 100 µm scale bar, lower right
            L = 100.0 / UM_PER_PX; W_, H_ = im.size
            ax.plot([W_ * 0.95 - L, W_ * 0.95], [H_ * 0.92] * 2, color="white", lw=2.2, solid_capstyle="butt")
            ax.text(W_ * 0.95 - L / 2, H_ * 0.88, "100 µm", color="white", fontsize=6.5, ha="center", va="bottom")
    import matplotlib.patheffects as pe
    axes[2].annotate("", xy=(0.30, 0.66), xytext=(0.60, 0.30), xycoords="axes fraction",
                     arrowprops=dict(arrowstyle="-|>", color="white", lw=1.4))
    axes[2].text(0.62, 0.27, "fractured edge", transform=axes[2].transAxes, ha="center", va="top", fontsize=7, color="white", fontweight="bold",
                 path_effects=[pe.withStroke(linewidth=1.6, foreground="#2b1d1d")])
    fig.subplots_adjust(left=0.01, right=0.99, top=0.90, bottom=0.11, wspace=0.04)
    fig.savefig(os.path.join(FIG, "Fig2_micrographs.png"), dpi=300); plt.close(fig)


# ------------------------------------------------------------------ Fig. 4 workflow schematic
def fig4():
    """Full-pipeline schematic (drawn, not data): experiments -> measurements -> signal processing ->
    sensor route and physics-informed few-shot route under leave-one-tool-out -> uncertainty -> evaluation."""
    from matplotlib.patches import Rectangle, Polygon, Circle
    BL, ORG, HELD, XW, YH = "#eaf2fb", C_FLEET, "#14213d", 126.0, 84.0
    fig = plt.figure(figsize=(6.3, 4.2)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, XW); ax.set_ylim(0, YH); ax.axis("off")
    rng = np.random.default_rng(7)

    def txt(x, y, s, fs=6.2, c=INK, **k):
        ax.text(x, y, s, fontsize=fs, color=c, va="center", linespacing=1.2, **k)

    def panel(x, y, w, h, ec="#8a8984", fc="white", ls="-", lw=0.9):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.2", fc=fc, ec=ec, lw=lw, ls=ls))

    def head(x, y, n, s, c=INK):
        ax.add_patch(Circle((x + 1.6, y), 1.55, fc=c, ec="none")); ax.text(x + 1.6, y, str(n), fontsize=6.4, color="white", ha="center", va="center", weight="bold")
        txt(x + 4.2, y, s, fs=7.0, c=c, weight="bold")

    def arr(x1, y1, x2, y2, c="#52514e"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", color=c, lw=0.9, shrinkA=0, shrinkB=0))

    def mini(x, y, w, h):
        m = fig.add_axes([x / XW, y / YH, w / XW, h / YH]); m.set_xticks([]); m.set_yticks([])
        for sp in ("top", "right"): m.spines[sp].set_visible(False)
        for sp in ("left", "bottom"): m.spines[sp].set_color("#8a8984"); m.spines[sp].set_linewidth(0.7)
        return m

    # =============== top row: data ===============
    # (1) experiments
    panel(1.5, 55, 34, 27); head(3, 79, 1, "Experiments")
    ax.add_patch(Rectangle((4, 64), 4.5, 7, fc="#d9d8d3", ec="#8a8984", lw=0.6))             # guide bush
    ax.add_patch(Rectangle((8.5, 66.6), 6, 1.8, fc="#b9b8b2", ec="#8a8984", lw=0.5))         # bar stock
    ax.add_patch(Polygon([(12.6, 65.2), (14.8, 66.3), (12.6, 67.4), (10.4, 66.3)], closed=True, fc=HELD, ec="none"))
    ax.add_patch(Polygon([(12.6, 65.2), (13.4, 62.6), (11.8, 62.6)], closed=True, fc="#8a8984", ec="none"))
    txt(17, 74.0, "Swiss-type turning", fs=6.0); txt(17, 70.8, "stainless 1.4571", fs=5.8, c=MUTED)
    txt(17, 67.6, "3 × 3 × 2 design", fs=6.0); txt(17, 64.4, "($v_\\mathrm{c}$, $f$, cooling)", fs=5.8, c=MUTED)
    txt(4, 60.4, "14 short tests (5 parts each)", fs=5.9); txt(4, 57.4, "4 long tests to breakage", fs=5.9)
    # (2) measurements
    panel(39, 55, 41, 27); head(40.5, 79, 2, "Measurements")
    t = np.linspace(0, 1, 300); w = np.sin(2 * np.pi * 40 * t) * rng.normal(1, 0.3, 300)
    ax.plot(42 + 8 * t, 72.5 + 1.8 * w, color="#52514e", lw=0.4)
    txt(52, 72.5, "vibration, axial + radial (50 kHz)", fs=6.0)
    ax.add_patch(Polygon([(46, 63.2), (49.6, 66.2), (46, 69.2), (42.4, 66.2)], closed=True, fc="#c9c8c2", ec="none"))
    ax.add_patch(Polygon([(46, 69.2), (47.0, 68.35), (45.0, 68.35)], closed=True, fc="#e3a07a", ec="none"))
    txt(52, 66.2, "VB$_{\\mathrm{max}}$ by microscope:\n104 values", fs=6.0)
    ax.plot(46, 59.8, "x", color="#e34948", ms=5, mew=1.3); txt(52, 59.8, "breakage noted: 4 tools", fs=6.0)
    # (3) signal processing
    panel(84, 55, 40.5, 27); head(85.5, 79, 3, "Signal processing")
    t = np.linspace(0, 1, 900); env = np.zeros_like(t)
    for c0 in np.linspace(0.08, 0.92, 6): env += np.exp(-((t - c0) / 0.035) ** 2)
    sig = np.sin(2 * np.pi * 160 * t) * (0.12 + env) * rng.normal(1, 0.25, 900)
    for c0 in np.linspace(0.08, 0.92, 6): ax.add_patch(Rectangle((87 + 34 * (c0 - 0.05), 66), 34 * 0.1, 10, fc=BL, ec="none"))
    ax.plot(87 + 34 * t, 71 + 3.6 * sig, color="#52514e", lw=0.35)
    txt(87, 62.0, "six contacts per part segmented;", fs=6.0); txt(87, 58.8, "294 time/frequency features", fs=6.0)
    arr(35.8, 68, 38.7, 68); arr(80.3, 68, 83.7, 68)

    # =============== bottom row: models under LOTO ===============
    panel(1.5, 1.5, 97, 47.5, ec=C_PHYS, ls="--")
    txt(63, 46.4, "Leave-one-tool-out folds", fs=6.6, c=C_PHYS, weight="bold"); txt(63, 43.6, "all fitting on the other tools only", fs=5.9, c=C_PHYS)
    # (4) sensor benchmark
    panel(4, 9, 26.5, 31, fc="#f4f3f0"); head(5, 37.3, 4, "Sensor benchmark")
    txt(6, 30.6, "features of a part →\nregressor → VB$_{\\mathrm{max}}$\nof the same part", fs=6.0)
    txt(6, 20.0, "ridge, LASSO, elastic net,\nrandom forest, XGBoost,\nMLP, PLS", fs=5.8, c=MUTED)
    txt(6, 12.0, "reads the current wear;\nno forecast", fs=5.8, c=MUTED)
    # (5) physics-informed few-shot
    panel(34, 9, 36, 31, fc="white", ec=C_PHYS); head(35, 37.3, 5, "Physics-informed few-shot", c=C_PHYS)
    a3 = mini(38.5, 17.2, 29, 16.0); tt = np.arange(1, 12); y = 70 + 26 * tt ** 0.5 + np.array([0, 3, -2, 2, -3, 1, 4, -2, 2, -1, 3])
    m = 3; xs = np.linspace(1, 11.5, 80); fit = 70 + 26 * xs ** 0.5
    a3.plot(xs, fit, color=C_PHYS, lw=1.4)
    a3.plot(tt[:m], y[:m], "o", color=HELD, ms=3.2); a3.plot(tt[m:], y[m:], "o", mfc="white", mec=INK, ms=2.8, mew=0.7)
    a3.axvline(m + 0.5, color="#8a8984", lw=0.6, ls=(0, (2, 2))); a3.set_xlim(0, 12.2); a3.set_ylim(70, 175)
    a3.set_xlabel("part number", fontsize=5.6, labelpad=1, color=MUTED); a3.set_ylabel("VB$_{\\mathrm{max}}$", fontsize=5.6, labelpad=1, color=MUTED)
    txt(36, 11.9, "VB = $b + a\\,t^{\\,p^*}$: $p^*$ from training tools;\n$b$, $a$ from the first $m$ inspections (●)", fs=5.8)
    # (6) uncertainty
    panel(73.5, 9, 22.5, 31, fc="white"); head(74.5, 37.3, 6, "Uncertainty")
    xs2 = np.linspace(0, 1, 50); ax.fill_between(76.5 + 16 * xs2, 24 + 7 * xs2 ** 0.6 - (1.5 + 4 * xs2), 24 + 7 * xs2 ** 0.6 + (1.5 + 4 * xs2), color="#d6e6f8", lw=0)
    ax.plot(76.5 + 16 * xs2, 24 + 7 * xs2 ** 0.6, color=C_PHYS, lw=1.1)
    txt(75.5, 14.2, "split-conformal\n90% band per horizon;\ntested tool not used\nin calibration", fs=5.6)
    # (7) evaluation
    panel(102, 22, 22.5, 27); head(103, 46.3, 7, "Evaluation")
    for j, (h, c) in enumerate(((0.5, C_PHYS), (0.95, C_FLEET), (0.75, C_LIN))):
        ax.add_patch(Rectangle((104 + j * 2.6, 33.5), 2.0, 8 * h, fc=c, ec="none"))
    txt(113, 37.5, "MAE, $R^2$\nper tool", fs=5.9)
    txt(104, 27.2, "coverage, band width;\nvs fleet-average, linear", fs=5.6, c=MUTED)
    # (8) breakage evidence
    panel(102, 1.5, 22.5, 17, fc="#fbefe9", ec=ORG); head(103, 15.6, 8, "Breakage", c=ORG)
    kx = np.array([0, 1.6, 1.6, 3.4, 3.4, 5, 5, 6.4]); ky = np.array([1, 1, 0.78, 0.78, 0.5, 0.5, 0.3, 0.3])
    ax.plot(104 + kx, 4.5 + 6 * ky, color=ORG, lw=1.0)
    txt(111.6, 7.6, "Kaplan–Meier,\nhazard (from 2);\nnot in forecasts", fs=5.5)
    # flows
    G = "#52514e"
    ax.plot([104, 104, 61.0], [54.7, 51.6, 51.6], color=G, lw=0.9); ax.plot([58.0, 17.2], [51.6, 51.6], color=G, lw=0.9); arr(17.2, 51.6, 17.2, 40.5)        # features -> sensor route
    arr(59.5, 54.7, 59.5, 40.5)                                                                         # VBmax -> few-shot route
    arr(70.3, 24, 73.2, 24); arr(96.3, 30, 101.7, 33)                                                  # few-shot -> uncertainty -> evaluation
    ax.plot([17.2, 17.2, 99.6, 99.6], [8.8, 5.0, 5.0, 26], color="#a8a7a2", lw=0.8, ls=(0, (3, 2))); arr(99.6, 26, 101.7, 26, c="#a8a7a2")
    fig.savefig(os.path.join(FIG, "Fig4_workflow.png"), dpi=300); plt.close(fig)


# ------------------------------------------------------------------ Fig. 5 sensor benchmark
def fig5():
    S = pd.read_csv(os.path.join(P2, "sensor_regression.csv"))
    S = S[(S.data == "measured labels only") & S.R2.notna()].copy()
    S["label"] = S.model.str.replace(" (alpha=10)", " (alpha = 10)", regex=False).str.replace("(CV alpha)", "(alpha by inner CV)", regex=False).str.replace("physics indicators", "wear indicators", regex=False).str.replace("relative to break-in", "relative to first part", regex=False)
    S = S.sort_values("R2")
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    y = np.arange(len(S))
    ax.hlines(y, S.R2, 0, color="#d9d8d3", lw=1.5)
    ax.scatter(S.R2, y, color=C_PHYS, s=26, zorder=3)
    for yy, (r2, mae) in zip(y, zip(S.R2, S.MAE)):
        ax.text(r2 - 0.03, yy, f"{r2:.2f}", ha="right", va="center", fontsize=6.5, color=INK)
    ax.axvline(0, color=INK, lw=0.8); ax.text(0.02, len(S) - 0.6, "$R^2$ = 0:\nmean predictor", fontsize=6.2, color=MUTED, va="top")
    ax.set_yticks(y); ax.set_yticklabels(S.label, fontsize=6.8); ax.set_xlim(-1.3, 0.3)
    ax.set_xlabel("Out-of-sample $R^2$ (leave-one-tool-out; 92 measured labels, 17 tools)"); ax.grid(axis="x")
    fig.savefig(os.path.join(FIG, "Fig5_sensor_benchmark.png"), bbox_inches="tight"); plt.close(fig)
    OUTD["fig5"] = S[["model", "R2", "MAE"]].to_dict("records")


# ------------------------------------------------------------------ Figs. 6-8 few-shot + conformal
def fig678():
    res3 = residuals(d, 3); summ3, q3 = conformal(res3, dd=d, m=3)
    res2 = residuals(d, 2); summ2, q2 = conformal(res2, dd=d, m=2)
    OUTD["conformal_m2"] = summ2; OUTD["conformal_m3"] = summ3
    # Fig 6
    fig, axs = plt.subplots(2, 2, figsize=(7.0, 4.8), sharey=True)
    titles = {"T14": "55 / 0.30 / coolant", "T16": "55 / 0.20 / coolant", "T17L": "55 / 0.08 / dry", "T18": "55 / 0.08 / coolant"}
    for ax, tt in zip(axs.ravel(), LONG):
        g = d[d.tool_id == tt].sort_values("order"); o, v = g.order.to_numpy(float), g.vb.to_numpy(float)
        grid = np.linspace(1, o.max(), 200)
        fp = forecast(d, tt, 3, "phys", grid); ff = forecast(d, tt, 3, "fleet", grid); fl = forecast(d, tt, 3, "linear", grid)
        qq = q3.get(tt, np.array([])); fut_o = o[3:]
        ph_f = forecast(d, tt, 3, "phys", fut_o)
        if len(qq):
            lo = ph_f - np.where(np.isfinite(qq), qq, np.nan); hi = ph_f + np.where(np.isfinite(qq), qq, np.nan)
            ax.fill_between(fut_o, lo, hi, color=C_PHYS, alpha=0.15, lw=0, step="mid", label="90 % conformal band")
        ax.plot(grid, fp, color=C_PHYS, lw=1.6, label="physics-informed law")
        ff_o = forecast(d, tt, 3, "fleet", o); ax.plot(o, ff_o, color=C_FLEET, lw=1.0, ls="--", marker="s", ms=2.5, label="fleet-average curve (at inspected parts)")
        ax.plot(grid, fl, color=C_LIN, lw=1.2, ls=":", label="linear extrapolation")
        ax.scatter(o[:3], v[:3], color=INK, s=16, zorder=4, label="adaptation window ($m$ = 3)")
        ax.scatter(o[3:], v[3:], facecolor="white", edgecolor=INK, s=16, zorder=4, label="later inspections (scored)")
        bp = D[(D.tool == tt) & D.breakage_row].part
        if len(bp): ax.axvline(float(bp.iloc[0]), color="#e34948", lw=0.9, ls=(0, (1, 1.5)), label="breakage recorded")
        mae = {k: float(np.mean(np.abs(forecast(d, tt, 3, k, fut_o) - v[3:]))) for k in ("phys", "fleet", "linear")}
        ax.set_title(f"{tt}: {titles[tt]}   MAE {mae['phys']:.0f} / {mae['fleet']:.0f} / {mae['linear']:.0f} µm", loc="left")
        ax.set_ylim(40, 320); ax.grid(axis="y"); ax.set_xlabel("Part number"); ax.set_ylabel("VB$_{\\mathrm{max}}$ (µm)")
        OUTD.setdefault("fig6", {})[tt] = mae
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=6.8, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig(os.path.join(FIG, "Fig6_long_run_forecasts.png")); plt.close(fig)

    # Fig 7 per-tool MAE
    P = pd.read_csv(os.path.join(P2, "fewshot_doe_pertool.csv"))
    names = {"physics base (Theil-Sen)": "Physics-informed", "fleet-average curve": "Fleet-average", "linear in t": "Linear"}
    P = P[P.model.isin(names)]; P["method"] = P.model.map(names)
    order = ["Physics-informed", "Fleet-average", "Linear"]; cols = [C_PHYS, C_FLEET, C_LIN]
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.8), sharey=True, gridspec_kw=dict(width_ratios=[1.3, 1, 1]))
    for ax, m in zip(axs, (2, 3, 4)):
        Pm = P[P.m == m]; tools = sorted(Pm.tool.unique())
        for t in tools:
            ys = [Pm[(Pm.tool == t) & (Pm.method == k)].MAE.values[0] for k in order]
            ax.plot(range(3), ys, color="#e6e5e1", lw=0.8, zorder=1)
        rng = np.random.default_rng(1)
        for i, (k, c) in enumerate(zip(order, cols)):
            vals = Pm[Pm.method == k].MAE.values
            ax.scatter(i + rng.uniform(-0.08, 0.08, len(vals)), vals, color=c, s=14, zorder=3, edgecolor="white", linewidth=0.4)
            ax.hlines(vals.mean(), i - 0.25, i + 0.25, color=INK, lw=1.6, zorder=4)
            ax.text(i + 0.28, vals.mean(), f"{vals.mean():.1f}", va="center", fontsize=6.5, color=INK)
        ax.set_xticks(range(3)); ax.set_xticklabels(order, rotation=20, fontsize=6.8)
        ax.set_title(f"$m$ = {m}: {len(tools)} tools" + (" (long runs)" if m > 2 else ""), loc="left"); ax.grid(axis="y")
    axs[0].set_ylabel("Per-tool MAE (µm)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "Fig7_accuracy_by_budget.png")); plt.close(fig)

    # Fig 8 coverage/width
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.8))
    for j, (summ, lab, mk, col) in enumerate(((summ2, "$m$ = 2 (17 tools)", "o", C_PHYS), (summ3, "$m$ = 3 (4 long runs)", "s", INK))):
        xs = np.arange(len(summ)) + (j - 0.5) * 0.22
        fin = np.array([s["mean_width"] is not None for s in summ])
        cov = np.array([s["coverage"] for s in summ]); ci = np.array([s["ci"] for s in summ]).T
        axs[0].errorbar(xs[fin], cov[fin], yerr=[(cov - ci[0])[fin], (ci[1] - cov)[fin]], fmt=mk, color=col, ms=4.5, capsize=2, lw=1, label=lab)
        for x in xs[~fin]:
            axs[0].plot(x, 0.36, mk, mfc="white", mec=MUTED, ms=4.5); axs[0].text(x, 0.40, "unbounded", rotation=90, fontsize=5.6, color=MUTED, ha="center", va="bottom")
        for x, s in zip(xs, summ):
            axs[0].text(x, 1.07, f"n={s['points']}", ha="center", fontsize=5.8, color=col)
        for x, s in zip(xs, summ):
            if s["mean_width"] is not None:
                axs[1].plot(x, s["mean_width"], mk, color=col, ms=4.5, label=lab if x == xs[-1] else None)
                axs[1].text(x, s["mean_width"] + 8, f"{s['mean_width']:.0f}", ha="center", fontsize=6, color=col)
            else:
                axs[1].text(x, 12, "unbounded", ha="center", fontsize=5.8, color=col, rotation=90, va="bottom")
    axs[0].axhline(0.9, color=MUTED, ls="--", lw=0.8)
    axs[0].set_ylim(0.3, 1.12); axs[0].set_ylabel("Empirical coverage (95 % CI)"); axs[1].set_ylabel("Mean band width (µm)")
    axs[1].set_ylim(0, 180)
    for ax in axs:
        ax.set_xticks(range(3)); ax.set_xticklabels(BINS); ax.set_xlabel("Forecast horizon (inspections ahead)"); ax.grid(axis="y")
    axs[0].text(1.45, 0.905, "target 0.90", fontsize=6, color=MUTED, ha="center", va="bottom"); fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=2, fontsize=6.5)
    fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig(os.path.join(FIG, "Fig8_coverage_width.png")); plt.close(fig)


# ------------------------------------------------------------------ Fig. 9 breakage evidence
def fig9():
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.9))
    cens = D[(D.regime == "short") & D.measured & (D.tool != "T17")].groupby("tool").vb.last().to_dict()
    for sc, t16, ls in (("S2", 154.0, "-"), ("S1", 587.0, "--")):
        ev = {"T14": 200.0, "T16": t16, "T18": 202.0, "T17L": 279.0}
        pts = [(v, 1) for v in ev.values()] + [(v, 0) for v in cens.values()]
        xs, ys, S = [0], [1.0], 1.0
        for w in sorted(set(v for v, e in pts if e)):
            n = sum(1 for v, e in pts if v >= w); dd = sum(1 for v, e in pts if v == w and e)
            xs += [w, w]; ys += [S, S * (1 - dd / n)]; S *= 1 - dd / n
        xs.append(620); ys.append(S)
        axs[0].plot(xs, ys, color=C_PHYS, ls=ls, lw=1.4, label=f"T16 = {t16:g} µm" + (" (post-damage)" if sc == "S2" else " (valid)"))
    axs[0].plot(list(cens.values()), [1.02] * len(cens), "|", color=MUTED, ms=7, label="censored short runs (last VB$_{\\mathrm{max}}$)")
    axs[0].set_xlim(0, 620); axs[0].set_ylim(0, 1.08); axs[0].set_xlabel("Last measured VB$_{\\mathrm{max}}$ (µm)")
    axs[0].set_ylabel("P(no breakage)"); axs[0].legend(fontsize=6.3, loc="lower left"); axs[0].grid(axis="y")
    axs[0].set_title("(a) Kaplan–Meier on the wear scale", loc="left")
    B = pd.read_csv(os.path.join(P2, "hazard_bands.csv"))
    for sc, ls in (("S2", "-"), ("S1", "--")):
        b = B[B.scenario == sc]
        axs[1].fill_between(b.vb, b.lo, b.hi, color=C_PHYS, alpha=0.12 if sc == "S2" else 0.0, lw=0)
        axs[1].plot(b.vb, b.h, color=C_PHYS, ls=ls, lw=1.4, label=f"fit, {'post-damage' if sc == 'S2' else 'valid'} coding")
        for c in [c for c in b.columns if c.startswith("loo_")]:
            axs[1].plot(b.vb, b[c], color="#c9c8c2", lw=0.7, zorder=0)
    ev_vb = [200, 202, 279, 154]
    axs[1].plot(ev_vb, [0.005] * 4, "x", color="#e34948", ms=6, label="breakage (last VB$_{\\mathrm{max}}$)")
    axs[1].axhline(0.10, color=MUTED, ls=":", lw=0.8); axs[1].text(52, 0.105, "h = 0.10 per part", fontsize=6, color=MUTED)
    axs[1].set_xlim(50, 400); axs[1].set_ylim(0, 0.6); axs[1].set_xlabel("VB$_{\\mathrm{max}}$ (µm)"); axs[1].set_ylabel("Per-part breakage hazard")
    axs[1].legend(fontsize=6.3, loc="upper left"); axs[1].grid(axis="y"); axs[1].set_title("(b) Exploratory discrete-time hazard", loc="left")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "Fig9_breakage_evidence.png")); plt.close(fig)


if __name__ == "__main__":
    fig2(); fig3(); fig4(); fig5(); fig678(); fig9()
    json.dump(OUTD, open(os.path.join(FIG, "figure_data.json"), "w"), indent=1, default=float)
    print(json.dumps(OUTD, indent=1, default=float))
