"""Compare GBM experiments: pooled metrics, firm-clustered paired bootstrap vs GB-01, round-trip check.

Bootstrap: same estimator as src/common/metrics.paired_bootstrap (resample whole firms with
replacement), implemented with firm multiplicity weights because the shared function builds its
firm index with an O(firms x rows) loop that is too slow on 506k rows. 200 draws (PLAYBOOK).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gbm_lib as G  # noqa: E402
from src.common import metrics as M  # noqa: E402

EXPS = ["GB-01", "GB-02", "GB-03", "GB-04", "GB-05", "GB-06"]


def wll(y, p, w):
    p = np.clip(p, M.EPS, 1 - M.EPS)
    return float(-np.sum(w * (y * np.log(p) + (1 - y) * np.log(1 - p))) / w.sum())


def boot(y, pn, po, firm, n_boot=200, seed=0):
    rng = np.random.default_rng(seed)
    firms, inv = np.unique(firm, return_inverse=True)
    d_auc, d_ll = [], []
    for _ in range(n_boot):
        cnt = np.bincount(rng.integers(0, len(firms), len(firms)), minlength=len(firms))
        w = cnt[inv].astype(float)
        m = w > 0
        d_auc.append(roc_auc_score(y[m], pn[m], sample_weight=w[m]) - roc_auc_score(y[m], po[m], sample_weight=w[m]))
        d_ll.append(wll(y[m], pn[m], w[m]) - wll(y[m], po[m], w[m]))
    return np.array(d_auc), np.array(d_ll)


def main():
    panel = pd.read_parquet(os.path.join(G.ROOT, "data/interim/panel.parquet"), columns=["row_id", "id", "y"])
    tr = pd.read_parquet(os.path.join(G.ROOT, "data/splits/walkforward_test_rows.parquet"))
    base = tr.merge(panel, on="row_id")
    P = {}
    for e in EXPS:
        o = pd.read_parquet(os.path.join(HERE, e, "oot_predictions.parquet"))
        assert len(o) == len(tr) and o["pd"].between(0, 1, inclusive="neither").all()
        assert o.merge(tr, on=["row_id", "fold"]).shape[0] == len(tr)
        P[e] = base.merge(o, on=["row_id", "fold"])["pd"].to_numpy()
    y, firm, yr = base["y"].astype(int).to_numpy(), base["id"].to_numpy(), base["test_year"].to_numpy()
    rows = []
    for e in EXPS:
        s = M.score(y, P[e], yr).set_index("group")
        seen = base["firm_seen_in_train"].to_numpy()
        rows.append(dict(exp=e, desc=G.EXPERIMENTS[e]["desc"], auc=s.loc["ALL", "auc"], ar=s.loc["ALL", "ar"],
                         log_loss=s.loc["ALL", "log_loss"], brier=s.loc["ALL", "brier"],
                         mean_pd=s.loc["ALL", "mean_pd"], observed_dr=s.loc["ALL", "observed_dr"],
                         cal_slope=s.loc["ALL", "cal_slope"],
                         auc_2008=s.loc[2008, "auc"], auc_2009=s.loc[2009, "auc"], auc_2010=s.loc[2010, "auc"],
                         auc_seen=M.auc(y[seen], P[e][seen]), auc_new=M.auc(y[~seen], P[e][~seen])))
    tab = pd.DataFrame(rows)
    brows = []
    for e in EXPS[1:]:
        da, dl = boot(y, P[e], P["GB-01"], firm)
        brows.append(dict(new=e, old="GB-01", d_auc=M.auc(y, P[e]) - M.auc(y, P["GB-01"]),
                          d_auc_ci=f"[{np.percentile(da, 2.5):+.4f}, {np.percentile(da, 97.5):+.4f}]",
                          share_auc_favours_new=float(np.mean(da > 0)),
                          d_log_loss=M.log_loss(y, P[e]) - M.log_loss(y, P["GB-01"]),
                          d_ll_ci=f"[{np.percentile(dl, 2.5):+.5f}, {np.percentile(dl, 97.5):+.5f}]",
                          share_ll_favours_new=float(np.mean(dl < 0))))
    bt = pd.DataFrame(brows)

    # round-trip: saved fold-3 model reproduces the fold-3 OOT predictions exactly
    art = G.load_model(os.path.join(HERE, "GB-01", "model_fold3"))
    full = pd.read_parquet(os.path.join(G.ROOT, "data/interim/panel.parquet"))
    te = full.merge(tr[tr["fold"] == 3][["row_id"]], on="row_id")
    p_rt = G.predict_pd(art, te)
    o3 = pd.read_parquet(os.path.join(HERE, "GB-01", "oot_predictions.parquet")).query("fold == 3")
    md = float(np.max(np.abs(te[["row_id"]].assign(p=p_rt).merge(o3, on="row_id").eval("p - pd"))))

    with open(os.path.join(HERE, "comparison.md"), "w") as fh:
        fh.write("# GBM experiments: comparison (pooled over test years 2008-2010, 506,159 rows, 7,784 defaults)\n\n"
                 "Own calibration in every row (Platt slope from inner slice B, level anchored to the training "
                 "default rate). AR = 2*AUC - 1. Brier = overall probability accuracy.\n\n")
        fh.write(tab.to_markdown(index=False, floatfmt=".5g"))
        fh.write("\n\n## Paired firm-clustered bootstrap vs GB-01 (200 draws; d = new - GB-01; "
                 "negative d_log_loss favours new)\n\n")
        fh.write(bt.to_markdown(index=False, floatfmt=".5g"))
        fh.write(f"\n\nRound-trip check: GB-01 fold-3 model saved with save_model and reloaded with load_model "
                 f"reproduces the fold-3 OOT PDs; max |diff| = {md:.2e}.\n")
    print(tab.to_string(index=False)); print(bt.to_string(index=False)); print("roundtrip maxdiff", md)


if __name__ == "__main__":
    main()
