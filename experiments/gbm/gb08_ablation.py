"""GB-08 step 1: leave-one-constraint-out ablation on the INNER split of each fold (no outer data used).

For fold k: inner-train -> early stopping on A, measured on A and on B (B untouched by fitting and stopping).
Runs: 'none_dropped' (= GB-01 inner model), one run per sign-constrained feature with only that constraint
dropped, and 'all_dropped' (= GB-02 inner model). GB-01's chosen cfg for the fold is used throughout.
cost_f = metric(constraint on) - metric(constraint f dropped): for log loss a POSITIVE cost means the
constraint makes the model worse on held-out inner data.
For every single-drop model the partial dependence (log-odds, 2,000 sampled A rows, grid = 2..98% quantiles
of inner-train + NaN) of the freed feature is stored, with the share of its movement against the expected sign.
Writes GB-08/ablation_inner.csv, GB-08/pd_curves.csv.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gbm_lib as G  # noqa: E402
import r2_lib as L  # noqa: E402
from src.common import metrics as M  # noqa: E402

OUT = os.path.join(HERE, "GB-08")
os.makedirs(OUT, exist_ok=True)


def pdp(bst, Xs, f, grid, n_iter):
    out = []
    for v in list(grid) + [np.nan]:
        X2 = Xs.copy()
        X2[f] = v
        out.append(float(np.mean(bst.predict(X2, num_iteration=n_iter, raw_score=True))))
    return np.array(out)


def against_share(curve, sign):
    d = np.diff(curve)
    tot = np.abs(d).sum()
    if tot == 0 or sign == 0:
        return np.nan
    return float((np.clip(-sign * d, 0, None)).sum() / tot)


def main():
    folds = [int(a) for a in sys.argv[1:]] or [3, 2, 1]
    panel = L.load_panel()
    wf, test_rows = L.load_splits()
    feats = L.feature_list("full")
    signs = L.B.expected_signs([f for f in feats if f in L.B.META])
    constrained = [f for f in feats if signs.get(f, 0) != 0]
    gb01 = json.load(open(os.path.join(HERE, "GB-01r", "config.json")))
    rows, curves = [], []
    for k in folds:
        f, tr, _ = L.fold_frames(panel, wf, test_rows, k)
        rec = [r for r in gb01["folds"] if r["fold"] == k][0]
        cfg = rec["chosen_cfg"]
        y = tr["y"].astype(int).to_numpy()
        m_tr, m_a, m_b, u, scheme = L.inner_masks(tr)
        fp_in = L.B.fit(tr[m_tr])
        Xtr, Xa, Xb = (G._X(tr[m], fp_in, feats, False) for m in (m_tr, m_a, m_b))
        Xs = Xa.sample(2000, random_state=0)
        runs = [("none_dropped", [])] + [(c, [c]) for c in constrained] + [("all_dropped", feats)]
        for name, relax in runs:
            t0 = time.time()
            mono = L.monotone_vector(feats, True, relax)
            bst = L.train_booster(Xtr, y[m_tr], cfg, mono, Xa, y[m_a])
            bi = int(bst.best_iteration)
            pa, pb = (np.clip(bst.predict(X, num_iteration=bi), G.EPS, 1 - G.EPS) for X in (Xa, Xb))
            r = dict(fold=k, scheme=scheme, dropped=name, best_iter=bi, A_ll=M.log_loss(y[m_a], pa),
                     B_ll=M.log_loss(y[m_b], pb), A_auc=M.auc(y[m_a], pa), B_auc=M.auc(y[m_b], pb),
                     AB_ll=M.log_loss(np.r_[y[m_a], y[m_b]], np.r_[pa, pb]),
                     AB_auc=M.auc(np.r_[y[m_a], y[m_b]], np.r_[pa, pb]))
            if name in constrained:
                v = Xtr[name].to_numpy(float)
                grid = np.unique(np.nanquantile(v[np.isfinite(v)], np.linspace(0.02, 0.98, 25)))
                cv = pdp(bst, Xs, name, grid, bi)
                r["pd_against_share"] = against_share(cv[:-1], signs[name])
                r["pd_range"] = float(cv[:-1].max() - cv[:-1].min())
                for g, c in zip(list(grid) + [np.nan], cv):
                    curves.append(dict(fold=k, feature=name, model="dropped", x=g, logodds=c))
            rows.append(r)
            print(f"fold {k} {name:22s} it={bi:4d} A_ll={r['A_ll']:.5f} B_ll={r['B_ll']:.5f} "
                  f"A_auc={r['A_auc']:.4f} B_auc={r['B_auc']:.4f} against={r.get('pd_against_share', np.nan):.2f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
            if name == "none_dropped":
                for c in constrained:
                    v = Xtr[c].to_numpy(float)
                    grid = np.unique(np.nanquantile(v[np.isfinite(v)], np.linspace(0.02, 0.98, 25)))
                    for g, cc in zip(list(grid) + [np.nan], pdp(bst, Xs, c, grid, bi)):
                        curves.append(dict(fold=k, feature=c, model="constrained", x=g, logodds=cc))
        df = pd.DataFrame(rows)
        base = df[(df.fold == k) & (df.dropped == "none_dropped")].iloc[0]
        for m in ("A_ll", "B_ll", "AB_ll"):
            df.loc[df.fold == k, f"cost_{m}"] = base[m] - df.loc[df.fold == k, m]
        for m in ("A_auc", "B_auc", "AB_auc"):
            df.loc[df.fold == k, f"cost_{m}"] = df.loc[df.fold == k, m] - base[m]
        rows = df.to_dict("records")
        tag = "_".join(str(x) for x in folds)
        pd.DataFrame(rows).to_csv(os.path.join(OUT, f"ablation_inner_f{tag}.csv"), index=False)
        pd.DataFrame(curves).to_csv(os.path.join(OUT, f"pd_curves_f{tag}.csv"), index=False)


if __name__ == "__main__":
    main()
