"""GB-08 step 2: summarise the inner ablation and plot the freed shapes of the costliest constraints.

Reads GB-08/ablation_inner_f3_2_1.csv and pd_curves_f3_2_1.csv (inner splits only).
Writes GB-08/ablation_summary.csv, GB-08/shapes_costly.png.
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "GB-08")
sys.path.insert(0, HERE)
import r2_lib as L  # noqa: E402


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "3_2_1"
    ab = pd.read_csv(os.path.join(OUT, f"ablation_inner_f{tag}.csv"))
    cv = pd.read_csv(os.path.join(OUT, f"pd_curves_f{tag}.csv"))
    signs = L.B.expected_signs()
    piv = ab.pivot_table(index="dropped", columns="fold", values=["cost_AB_ll", "cost_AB_auc", "pd_against_share"])
    piv.columns = [f"{a}_f{b}" for a, b in piv.columns]
    piv["mean_cost_AB_ll"] = piv[[c for c in piv.columns if c.startswith("cost_AB_ll")]].mean(axis=1)
    piv["mean_cost_AB_auc"] = piv[[c for c in piv.columns if c.startswith("cost_AB_auc")]].mean(axis=1)
    piv["n_folds_costly_ll"] = (piv[[c for c in piv.columns if c.startswith("cost_AB_ll_f")]] > 0).sum(axis=1)
    piv["expected_sign"] = [signs.get(i, np.nan) for i in piv.index]
    piv = piv.sort_values("mean_cost_AB_ll", ascending=False)
    piv.to_csv(os.path.join(OUT, "ablation_summary.csv"))
    print(piv.round(5).to_string())
    feats = [f for f in piv.index if f in signs][:9]
    fig, ax = plt.subplots(3, 3, figsize=(15, 11))
    for a, f in zip(ax.ravel(), feats):
        for k, col in ((1, "tab:green"), (2, "tab:orange"), (3, "tab:blue")):
            for model, ls in (("dropped", "-"), ("constrained", ":")):
                c = cv[(cv.feature == f) & (cv.fold == k) & (cv.model == model)]
                c = c[c.x.notna()]
                if len(c):
                    a.plot(c.x, c.logodds - c.logodds.mean(), ls=ls, color=col,
                           label=f"fold {k} {'free' if model == 'dropped' else 'constrained'}")
        a.set_title(f"{f} (expected sign {signs[f]:+d}); mean inner cost {piv.loc[f, 'mean_cost_AB_ll']:+.5f}", fontsize=9)
        a.set_ylabel("partial dependence, log-odds (centred)")
    ax[0, 0].legend(fontsize=7)
    fig.suptitle("GB-08: shape of each feature when ONLY its constraint is dropped (solid) vs GB-01 (dotted); inner splits")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "shapes_costly.png"), dpi=100)


if __name__ == "__main__":
    main()
