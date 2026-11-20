"""Round-2 inner ablation with seed noise (feature-architect, 2026-11-02).

Run:  OMP_NUM_THREADS=3 python -m src.features.ablation_v2
Inner split only: fit params + LightGBM on fs_year 2007-08, score 2009 (panel read with a parquet
filter fs_year <= 2009; LOCKBOX respected). Fixed LightGBM settings (no tuning), 3 seeds per set.
Writes reports/features/round2_ablation_seeds.csv. NOTE: the run of 2026-11-02 15:11 used the
21 candidates ("full2" in that csv = all 21) with interest_to_sales constrained "+"; the final-set
check (45 features, interest_to_sales unconstrained) is in round2_lgb_ablation.csv.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from src.common.metrics import auc, log_loss
from src.features import build as B
from src.features.univariate_v2 import OUT, load_train

SEEDS = [0, 1, 2]


def run(Xa, ya, Xb, yb, cols, signs, seed):
    import lightgbm as lgb
    mono = [int(signs.get(c, 0)) if c not in B.CAT_FEATURES else 0 for c in cols]
    m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=200,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                           monotone_constraints=mono, n_jobs=3, verbose=-1, random_state=seed)
    m.fit(Xa[cols], ya)
    p = m.predict_proba(Xb[cols])[:, 1]
    return auc(yb, p), log_loss(yb, p)


def main():
    tr = load_train()
    a, b = tr["fs_year"] <= 2008, tr["fs_year"] == 2009
    tra, trb = tr[a].reset_index(drop=True), tr[b].reset_index(drop=True)
    pa = B.fit(tra, feature_set="full2")
    Xa = B.transform(tra, pa, feature_set="full2_candidates", oof=True)
    Xb = B.transform(trb, pa, feature_set="full2_candidates")
    ya, yb = tra["y"].astype(float), trb["y"].astype(float)
    signs = B.expected_signs(B.ALL_FEATURES + B.V2_CANDIDATES)
    v1, f2 = B.ALL_FEATURES, B.ALL_FEATURES + B.V2_CANDIDATES   # ran with all 21 candidates
    sets = {"v1 full": v1, "full2": f2}
    for c in B.V2_CANDIDATES:
        sets[f"v1 + {c}"] = v1 + [c]
    sets["full2 - sector pct block"] = [c for c in f2 if c not in B.V2_PCT]
    for c in ["net_debt_coverage", "n_missing_raw", "payables_to_sales", "tfr_ratio",
              "personnel_to_sales", "province_te", "region_te", "n_core_imputed"]:
        sets[f"full2 - {c}"] = [x for x in f2 if x != c]
    rows = []
    for name, cols in sets.items():
        for s in SEEDS:
            au, ll = run(Xa, ya, Xb, yb, cols, signs, s)
            rows.append(dict(set=name, seed=s, n_features=len(cols), auc_2009=au, logloss_2009=ll))
        print(name, np.mean([r["auc_2009"] for r in rows[-3:]]), flush=True)
    df = pd.DataFrame(rows)
    g = df.groupby("set", sort=False).agg(n_features=("n_features", "first"), auc_mean=("auc_2009", "mean"),
                                          auc_sd=("auc_2009", "std"), ll_mean=("logloss_2009", "mean"),
                                          ll_sd=("logloss_2009", "std")).reset_index()
    g["d_auc_vs_v1"] = g["auc_mean"] - g.loc[g.set == "v1 full", "auc_mean"].iloc[0]
    g["d_ll_vs_v1"] = g["ll_mean"] - g.loc[g.set == "v1 full", "ll_mean"].iloc[0]
    g["d_auc_vs_full2"] = g["auc_mean"] - g.loc[g.set == "full2", "auc_mean"].iloc[0]
    g["d_ll_vs_full2"] = g["ll_mean"] - g.loc[g.set == "full2", "ll_mean"].iloc[0]
    g.to_csv(os.path.join(OUT, "round2_ablation_seeds.csv"), index=False)
    df.to_csv(os.path.join(OUT, "round2_ablation_seeds_raw.csv"), index=False)
    print(g.to_markdown(index=False, floatfmt=".5f"))


if __name__ == "__main__":
    main()
