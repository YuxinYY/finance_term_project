"""Round-2 comparison vs GB-01 (incumbent). Paired firm-clustered bootstrap (referee's fast engine copied
in r2_lib, 200 draws, seed 0) on the pooled test set and per test year.

WIN RULE (orchestrator, round 2): pooled paired CI of dAUC excludes 0 in the favourable direction AND the
variant's AUC is higher in each of 2008, 2009, 2010. Same rule reported for log loss separately.

usage: python experiments/gbm/compare_r2.py GB-08 GB-09 ...   (writes experiments/gbm/comparison_r2.csv/.md)
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import r2_lib as L  # noqa: E402
from src.common import metrics as M  # noqa: E402

LOC = {"GB-01": "GB-01", "GB-02": "GB-02"}


def main():
    exps = sys.argv[1:]
    wf, tr = L.load_splits()
    pan = pd.read_parquet(os.path.join(L.ROOT, "data/interim/panel.parquet"), columns=["row_id", "id", "y", "fs_year"],
                          filters=[("fs_year", "<", L.LOCKBOX_YEAR)])
    base = tr.merge(pan, on="row_id", how="left", validate="one_to_one")
    y, firm, yr = base["y"].astype(int).to_numpy(), base["id"].to_numpy(), base["test_year"].to_numpy()
    seen = base["firm_seen_in_train"].to_numpy(bool)
    P = {}
    for e in ["GB-01", "GB-02"] + exps:
        o = pd.read_parquet(os.path.join(HERE, e, "oot_predictions.parquet"))
        p = base[["row_id", "fold"]].merge(o, on=["row_id", "fold"], how="left")["pd"].to_numpy(float)
        assert np.isfinite(p).all() and len(p) == len(base), e
        P[e] = p
    subsets = {"ALL": np.ones(len(y), bool), "2008": yr == 2008, "2009": yr == 2009, "2010": yr == 2010}
    draws = L.bootstrap_draws(y, firm, P, subsets, n_boot=200, seed=0)
    rows, prow = [], []
    for e, p in P.items():
        r = {"exp": e}
        for s, m in subsets.items():
            cs = M.calibration_slope_intercept(y[m], p[m])
            r[f"auc_{s}"], r[f"ll_{s}"], r[f"slope_{s}"] = M.auc(y[m], p[m]), M.log_loss(y[m], p[m]), cs["cal_slope"]
        r["auc_seen"], r["auc_new"] = M.auc(y[seen], p[seen]), M.auc(y[~seen], p[~seen])
        r["mean_pd"], r["brier"] = float(p.mean()), float(np.mean((p - y) ** 2))
        rows.append(r)
        if e == "GB-01":
            continue
        q = {"new": e, "old": "GB-01"}
        for s, m in subsets.items():
            da = draws[("auc", e, s)] - draws[("auc", "GB-01", s)]
            dl = draws[("log_loss", e, s)] - draws[("log_loss", "GB-01", s)]
            q[f"dAUC_{s}"] = M.auc(y[m], p[m]) - M.auc(y[m], P["GB-01"][m])
            q[f"dAUC_{s}_ci"] = "[%+.4f, %+.4f]" % L.ci(da)
            q[f"dLL_{s}"] = M.log_loss(y[m], p[m]) - M.log_loss(y[m], P["GB-01"][m])
            q[f"dLL_{s}_ci"] = "[%+.5f, %+.5f]" % L.ci(dl)
            if s == "ALL":
                q["auc_ci_excl0"], q["ll_ci_excl0"] = L.ci(da)[0] > 0, L.ci(dl)[1] < 0
        q["auc_better_all_3_years"] = all(q[f"dAUC_{t}"] > 0 for t in ("2008", "2009", "2010"))
        q["ll_better_all_3_years"] = all(q[f"dLL_{t}"] < 0 for t in ("2008", "2009", "2010"))
        q["WIN_auc"] = bool(q["auc_ci_excl0"] and q["auc_better_all_3_years"])
        q["WIN_ll"] = bool(q["ll_ci_excl0"] and q["ll_better_all_3_years"])
        prow.append(q)
    tab, pb = pd.DataFrame(rows), pd.DataFrame(prow)
    # benchmark pairs: is the gap to the RF benchmark real, and how does the unconstrained boosted twin compare?
    bench = []
    if "GB-11" in P:
        for new, old in (("GB-11", "GB-01"), ("GB-02", "GB-11"), ("GB-08", "GB-11")):
            if new not in P:
                continue
            for s, m in subsets.items():
                da = draws[("auc", new, s)] - draws[("auc", old, s)]
                dl = draws[("log_loss", new, s)] - draws[("log_loss", old, s)]
                bench.append(dict(new=new, old=old, subset=s, dAUC=M.auc(y[m], P[new][m]) - M.auc(y[m], P[old][m]),
                                  dAUC_ci="[%+.4f, %+.4f]" % L.ci(da),
                                  dLL=M.log_loss(y[m], P[new][m]) - M.log_loss(y[m], P[old][m]),
                                  dLL_ci="[%+.5f, %+.5f]" % L.ci(dl)))
    bench = pd.DataFrame(bench)
    bench.to_csv(os.path.join(HERE, "comparison_r2_benchmark_pairs.csv"), index=False)
    yr_rows = []
    for _, r in pb.iterrows():
        for t in ("2008", "2009", "2010"):
            yr_rows.append(dict(new=r["new"], year=t, dAUC=r[f"dAUC_{t}"], dAUC_ci=r[f"dAUC_{t}_ci"],
                                dLL=r[f"dLL_{t}"], dLL_ci=r[f"dLL_{t}_ci"]))
    yr = pd.DataFrame(yr_rows)
    tab.to_csv(os.path.join(HERE, "comparison_r2_metrics.csv"), index=False)
    pb.to_csv(os.path.join(HERE, "comparison_r2_bootstrap.csv"), index=False)
    cols = ["exp", "auc_ALL", "auc_2008", "auc_2009", "auc_2010", "ll_ALL", "ll_2008", "ll_2009", "ll_2010",
            "slope_ALL", "slope_2008", "slope_2009", "slope_2010", "auc_seen", "auc_new", "mean_pd", "brier"]
    c2 = ["new", "dAUC_ALL", "dAUC_ALL_ci", "dAUC_2008", "dAUC_2009", "dAUC_2010", "dLL_ALL", "dLL_ALL_ci",
          "dLL_2008", "dLL_2009", "dLL_2010", "WIN_auc", "WIN_ll"]
    with open(os.path.join(HERE, "comparison_r2.md"), "w") as fh:
        fh.write("# Round-2 GBM comparison (own calibration; pooled = 2008-2010, 506,159 rows, 7,784 defaults)\n\n")
        fh.write("Generated by `experiments/gbm/compare_r2.py`. slope = logistic recalibration slope (1 = perfect).\n\n")
        fh.write(tab[cols].to_markdown(index=False, floatfmt=".5f"))
        fh.write("\n\n## Paired firm-clustered bootstrap vs GB-01 (200 draws; d = new - GB-01)\n\n")
        fh.write(pb[c2].to_markdown(index=False, floatfmt="+.5f"))
        fh.write("\n\nWIN = pooled CI excludes 0 AND point estimate better in all 3 test years (no multiplicity correction).\n")
        fh.write("\n## Per-year paired differences vs GB-01\n\n")
        fh.write(yr.to_markdown(index=False, floatfmt="+.5f"))
        if len(bench):
            fh.write("\n\n## Benchmark pairs (random forest GB-11)\n\n")
            fh.write(bench.to_markdown(index=False, floatfmt="+.5f"))
        fh.write("\n")
    # keep round 1 in comparison.md; replace only the marked round-2 section
    cmp_path, mark = os.path.join(HERE, "comparison.md"), "<!-- ROUND-2 PHASE A (compare_r2.py) -->"
    old = open(cmp_path).read().split(mark)[0].rstrip()
    with open(cmp_path, "w") as fh:
        fh.write(old + "\n\n" + mark + "\n\n" + open(os.path.join(HERE, "comparison_r2.md")).read())
    print(tab[cols].round(4).to_string(index=False))
    print(pb[c2].to_string(index=False))
    print(yr.to_string(index=False))
    print(bench.to_string(index=False))
    # row-level reproduction check GB-01r vs GB-01
    if "GB-01r" in P:
        print("GB-01r vs GB-01 max abs PD diff:", float(np.max(np.abs(P["GB-01r"] - P["GB-01"]))))


if __name__ == "__main__":
    main()
