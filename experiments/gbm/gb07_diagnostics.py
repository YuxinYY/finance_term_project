"""GB-07: overfitting diagnostics for GB-01 (and its unconstrained twin GB-02 as contrast).

Per fold:
  * inner model (GB-01's chosen cfg, trained on inner-train): AUC on inner-train (in-sample), on A (early
    stopping slice) and on B (untouched); learning curve: train and A log loss for 1..N trees, no early stop.
  * final model: in-sample AUC/AR on the whole training window vs outer out-of-time AUC/AR (from GB-01r).
  * calibration slope/intercept per test year (from GB-01r metrics_by_year.csv).
Outer test labels are NOT used for any curve (only the already-scored GB-01 / GB-02 OOT numbers are quoted).
Writes experiments/gbm/GB-07/{gap_by_fold.csv, learning_curve.csv, calibration_by_year.csv, learning_curve.png,
gap.png, summary.md}.
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gbm_lib as G  # noqa: E402
import r2_lib as L  # noqa: E402
from src.common import metrics as M  # noqa: E402

OUT = os.path.join(HERE, "GB-07")
os.makedirs(OUT, exist_ok=True)


def main():
    panel = L.load_panel()
    wf, test_rows = L.load_splits()
    feats = L.feature_list("full")
    gb01 = json.load(open(os.path.join(HERE, "GB-01r", "config.json")))
    gb02 = json.load(open(os.path.join(HERE, "GB-02", "config.json")))
    m02 = pd.read_csv(os.path.join(HERE, "GB-02", "metrics_by_year.csv")).set_index("group")
    gap, curves = [], []
    for k in (1, 2, 3):
        f, tr, te = L.fold_frames(panel, wf, test_rows, k)
        rec = [r for r in gb01["folds"] if r["fold"] == k][0]
        y = tr["y"].astype(int).to_numpy()
        m_tr, m_a, m_b, u, _ = L.inner_masks(tr)
        fp_in = L.B.fit(tr[m_tr])
        Xtr, Xa, Xb = (G._X(tr[m], fp_in, feats, False) for m in (m_tr, m_a, m_b))
        for name, relax in (("GB-01", []), ("GB-02", feats)):
            mono = L.monotone_vector(feats, True, relax)
            cfg = rec["chosen_cfg"] if name == "GB-01" else [r for r in gb02["folds"] if r["fold"] == k][0]["chosen_cfg"]
            bi = rec["best_iter"] if name == "GB-01" else [r for r in gb02["folds"] if r["fold"] == k][0]["best_iter"]
            n_max = max(1500, 3 * bi)
            ev = {}
            bst = L.train_booster(Xtr, y[m_tr], cfg, mono, Xa, y[m_a], w=None, record=ev,
                                  max_rounds=n_max, early_stop=None)
            if name == "GB-01":
                curves.append(pd.DataFrame({"fold": k, "trees": np.arange(1, n_max + 1),
                                            "train_log_loss": ev["train"]["binary_logloss"],
                                            "A_log_loss": ev["valid"]["binary_logloss"], "best_iter": bi}))
            a_in = M.auc(y[m_tr], bst.predict(Xtr, num_iteration=bi))
            a_a = M.auc(y[m_a], bst.predict(Xa, num_iteration=bi))
            a_b = M.auc(y[m_b], bst.predict(Xb, num_iteration=bi))
            # final-model in-sample (refit on the full window with the fold's chosen cfg and tree count)
            if name == "GB-01":
                ins, oot = rec["in_sample"]["auc"], rec["oot_auc"]
            else:
                fp = L.B.fit(tr)
                Xall = G._X(tr, fp, feats, False)
                n_final = int(round(G.N_FINAL_FACTOR * bi))
                b2 = L.train_booster(Xall, y, cfg, mono, n_rounds=n_final)
                ins = M.auc(y, b2.predict(Xall))
                oot = float(m02.loc[str(f["test_year"]), "auc"])
            gap.append(dict(model=name, fold=k, train_years=str(f["train_years"]), test_year=f["test_year"],
                            inner_train_auc=a_in, inner_A_auc=a_a, inner_B_auc=a_b,
                            inner_gap_auc_train_minus_B=a_in - a_b,
                            final_in_sample_auc=ins, oot_auc=oot, gap_auc=ins - oot,
                            final_in_sample_ar=2 * ins - 1, oot_ar=2 * oot - 1,
                            gap_ar_points=100 * ((2 * ins - 1) - (2 * oot - 1))))
            print(gap[-1], flush=True)
    gap = pd.DataFrame(gap)
    gap.to_csv(os.path.join(OUT, "gap_by_fold.csv"), index=False)
    cur = pd.concat(curves, ignore_index=True)
    cur.to_csv(os.path.join(OUT, "learning_curve.csv"), index=False)
    cal = pd.read_csv(os.path.join(HERE, "GB-01r", "metrics_by_year.csv"))
    cal = cal[["group", "n", "defaults", "auc", "log_loss", "mean_pd", "observed_dr", "cal_intercept", "cal_slope"]]
    cal.to_csv(os.path.join(OUT, "calibration_by_year.csv"), index=False)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for i, k in enumerate((1, 2, 3)):
        c = cur[cur.fold == k]
        bi = int(c.best_iter.iloc[0])
        a = ax[i]
        a.plot(c.trees, c.A_log_loss, label="inner validation A (held out)")
        a2 = a.twinx()
        a2.plot(c.trees, c.train_log_loss, color="grey", ls="--", label="inner train")
        a.axvline(bi, color="k", lw=0.8)
        a.axvline(1.2 * bi, color="r", lw=0.8, ls=":")
        lo = c.A_log_loss.min()
        a.set_ylim(lo - 0.0005, lo + 0.004)
        a.set_title(f"fold {k}: best_iter {bi} (black), final 1.2x (red)")
        a.set_xlabel("trees")
        a.set_ylabel("A log loss")
        a2.set_ylabel("train log loss (grey)")
    fig.suptitle("GB-01 learning curves on the inner split (no early stopping)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "learning_curve.png"), dpi=110)

    fig, ax = plt.subplots(figsize=(7, 4))
    for name, mk in (("GB-01", "o"), ("GB-02", "s")):
        g = gap[gap.model == name]
        ax.plot(g.fold, g.final_in_sample_auc, marker=mk, ls="-", label=f"{name} in-sample (train window)")
        ax.plot(g.fold, g.oot_auc, marker=mk, ls="--", label=f"{name} out-of-time (next year)")
    ax.set_xticks([1, 2, 3])
    ax.set_xlabel("fold (test year 2008 / 2009 / 2010)")
    ax.set_ylabel("AUC")
    ax.legend(fontsize=8)
    ax.set_title("In-sample vs out-of-time AUC")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "gap.png"), dpi=110)

    # learning-curve flatness: A log loss at 1.2x best vs best, and at 2x best
    flat = []
    for k in (1, 2, 3):
        c = cur[cur.fold == k].set_index("trees")
        bi = int(c.best_iter.iloc[0])
        flat.append(dict(fold=k, best_iter=bi, A_ll_best=c.A_log_loss[bi],
                         A_ll_1p2x=c.A_log_loss[int(round(1.2 * bi))], A_ll_2x=c.A_log_loss[min(2 * bi, c.index.max())],
                         A_ll_max=c.A_log_loss.iloc[-1], trees_max=int(c.index.max()),
                         train_ll_best=c.train_log_loss[bi], train_ll_max=c.train_log_loss.iloc[-1]))
    flat = pd.DataFrame(flat)
    flat.to_csv(os.path.join(OUT, "learning_curve_flatness.csv"), index=False)
    with open(os.path.join(OUT, "summary.md"), "w") as fh:
        fh.write("# GB-07 overfitting diagnostics (GB-01; GB-02 as contrast)\n\n## AUC gaps\n\n")
        fh.write(gap.to_markdown(index=False, floatfmt=".4f"))
        fh.write("\n\n## Learning-curve flatness (inner slice A log loss)\n\n")
        fh.write(flat.to_markdown(index=False, floatfmt=".5f"))
        fh.write("\n\n## Calibration per test year (GB-01, own calibration)\n\n")
        fh.write(cal.to_markdown(index=False, floatfmt=".4f"))
        fh.write("\n")
    print(gap.to_string()); print(flat.to_string()); print(cal.to_string())


if __name__ == "__main__":
    main()
