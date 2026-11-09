"""Calibration curves for Chloe's LightGBM, round 1 (test FY2009, her label).

Reads 2026-11-08_fy2009_predictions.parquet (written by 2026-11-08_r1_fit_and_calibrate.py);
writes three PNGs to reports/calibration/fig/ and the binned numbers behind fig 1 to a CSV.
Nothing is fitted here: it only plots PDs that were already produced without FY2009 labels.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import logit

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FIG = ROOT / "reports/calibration/fig"
FIG.mkdir(parents=True, exist_ok=True)

d = pd.read_parquet(HERE / "2026-11-08_fy2009_predictions.parquet")
y = d["y_chloe"].to_numpy()

INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dc"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False})

VARIANTS = [
    ("pd_raw", "1. Raw LightGBM (no calibration)", BLUE),
    ("pd_b_platt", "2. Chloe's Platt (fit on FY2008)", ORANGE),
    ("pd_iso_val_e", "3. Isotonic + level shift", YELLOW),
    ("pd_d_ours_e", "4. Ours: prior shift to rule e", AQUA),
]
TICKS = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]
TLAB = ["0.1%", "0.3%", "1%", "3%", "10%", "30%"]

# ---------- Fig 1: reliability diagrams (20 equal-size groups by PD) ----------
rows = []
fig, axes = plt.subplots(1, 4, figsize=(15, 4.2), sharex=True, sharey=True)
for ax, (col, name, color) in zip(axes, VARIANTS):
    p = d[col].to_numpy()
    grp = pd.qcut(p, 20, labels=False, duplicates="drop")  # bins by PD value; tied PDs stay together
    b = pd.DataFrame({"p": p, "y": y, "g": grp}).groupby("g").agg(mean_pd=("p", "mean"),
                                                                     obs=("y", "mean"), n=("y", "size"))
    b["variant"] = col
    rows.append(b.reset_index())
    lo, hi = 3e-4, 0.6
    ax.plot([lo, hi], [lo, hi], ls="--", lw=1, color=MUTED)
    ax.plot(b.mean_pd, b.obs.clip(lower=3e-4), "-o", lw=2, ms=6, color=color,
            markeredgecolor="white", markeredgewidth=1.5)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xticks(TICKS, TLAB); ax.set_yticks(TICKS, TLAB); ax.minorticks_off()
    ax.grid(color=GRID, lw=0.8)
    ratio = y.mean() / p.mean()
    ax.set_title(f"{name}\nmean PD {p.mean():.2%} vs observed {y.mean():.2%}  (obs/pred {ratio:.2f})",
                 fontsize=9.5, color=INK, loc="left")
    ax.set_xlabel("Predicted PD (group average)")
axes[0].set_ylabel("Observed default rate")
axes[0].text(0.012, 0.0006, "on the dashed line =\nperfectly calibrated", fontsize=8, color=MUTED)
axes[0].text(0.0004, 0.09, "dots ABOVE the line =\nPDs too low", fontsize=8, color=MUTED)
fig.suptitle("Calibration curves, FY2009 test year (170,549 firms, up to 20 equal-size PD groups), Chloe's LightGBM round 1",
             x=0.01, ha="left", fontsize=11.5, color=INK)
fig.tight_layout()
fig.savefig(FIG / "2026-11-08_chloe_r1_reliability_curves.png", dpi=150)
pd.concat(rows).to_csv(HERE / "2026-11-08_reliability_bins_fy2009.csv", index=False)

# ---------- Fig 2: what each method does to a PD (raw -> calibrated) ----------
fig, ax = plt.subplots(figsize=(7.2, 5.6))
s = d.sort_values("pd_raw")
lo, hi = 3e-4, 0.6
ax.plot([lo, hi], [lo, hi], ls="--", lw=1, color=MUTED)
ax.text(0.07, 0.045, "no change", color=MUTED, fontsize=8.5, rotation=38)
for col, name, color in VARIANTS[1:]:
    ax.plot(s.pd_raw, s[col], lw=2, color=color, label=name[3:])
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
ax.set_xticks(TICKS, TLAB); ax.set_yticks(TICKS, TLAB); ax.minorticks_off()
ax.grid(color=GRID, lw=0.8)
ax.set_xlabel("PD before calibration (raw LightGBM)")
ax.set_ylabel("PD after calibration")
ax.legend(frameon=False, loc="upper left", fontsize=9)
ax.set_title("What each method does to one firm's PD\n"
             "Prior shift and Platt: smooth curves just above the line. Isotonic: a staircase (ties).",
             loc="left", fontsize=10, color=INK)
fig.tight_layout()
fig.savefig(FIG / "2026-11-08_chloe_r1_mapping_curves.png", dpi=150)

# ---------- Fig 3: prior shift = slide every firm right by the same amount ----------
z0, z1 = logit(d.pd_raw.clip(1e-6, 1 - 1e-6)), logit(d.pd_d_ours_e.clip(1e-6, 1 - 1e-6))
delta = float(np.median(z1 - z0))
fig, ax = plt.subplots(figsize=(8.5, 4.4))
bins = np.linspace(-9.5, 1.5, 90)
ax.hist(z0, bins=bins, histtype="step", lw=2, color=BLUE, label=f"before: mean PD {d.pd_raw.mean():.2%}")
ax.hist(z1, bins=bins, histtype="step", lw=2, color=AQUA, label=f"after prior shift: mean PD {d.pd_d_ours_e.mean():.2%}")
pt = [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 0.7]
ax.set_xticks(logit(np.array(pt)), ["0.03%", "0.1%", "0.3%", "1%", "3%", "10%", "30%", "70%"])
ax.set_xlabel("Firm's PD (log-odds scale: equal steps = equal multiplication of the odds)")
ax.set_ylabel("Number of firms")
ax.annotate("", xy=(-1.0 + delta, 3600), xytext=(-1.0, 3600),
            arrowprops=dict(arrowstyle="->", color=INK, lw=1.5))
ax.text(-0.9, 3850, f"every firm moves right by the same δ = {delta:.2f}\n(odds × {np.exp(delta):.2f})",
        fontsize=9, color=INK)
ax.legend(frameon=False, loc="upper right", fontsize=9)
ax.grid(axis="y", color=GRID, lw=0.8)
ax.set_title("Prior shift: same shape, same order, whole distribution slides up. Observed FY2009 rate: "
             f"{y.mean():.2%}", loc="left", fontsize=10, color=INK)
fig.tight_layout()
fig.savefig(FIG / "2026-11-08_chloe_r1_prior_shift_slide.png", dpi=150)
print("delta", delta, "odds x", np.exp(delta))
print(pd.concat(rows).groupby("variant").size())
