"""Estimation step 2: train the final LightGBM on all FY2007-2011 labelled rows and freeze every parameter the
harness needs.

Model (KN decision 2026-11-21: unconstrained; D-158 grid winner, same setting in both modes):
  num_leaves 31, min_child_samples 3000, feature_fraction 0.5, bagging off, lambda_l2 10, learning_rate 0.05,
  fixed 210 trees (round-3 best iterations for this setting: 205 refit, 205-215 over seeds; no early stopping
  is possible without a validation set), full-data histogram bins, deterministic.
Level (D-072 prior shift + D-075 floor/cap; tau from experiments/calibration/2026-11-08_chloe_lgbm_v1/
  2026-11-08_params_for_production.json, 5-month label, D-151):
  PD = clip(expit(a + logit(p_raw) + delta), 0.0003, 0.99),
  a solved so the mean in-sample PD equals the training default rate, delta = logit(tau) - logit(train rate).
Outputs (in this folder): final_lightgbm.txt, final_params.json, final_insample_summary.json
"""
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score, log_loss

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "vendor_7949207" / "src"))
from finance_project.data_processing import preprocess  # noqa: E402,F401  (needed to unpickle artifacts)

REPO = HERE.parents[2]
CAL = json.loads((REPO / "experiments/calibration/2026-11-08_chloe_lgbm_v1/2026-11-08_params_for_production.json").read_text())
TAU, FLOOR, CAP = CAL["tau"], CAL["floor"], CAL["cap"]
N_TREES = 210
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_child_samples=3000,
              feature_fraction=0.5, bagging_fraction=1.0, bagging_freq=0, lambda_l2=10.0, max_bin=255,
              bin_construct_sample_cnt=10**7, feature_pre_filter=False, seed=42, deterministic=True,
              force_row_wise=True, num_threads=8, verbose=-1)


def main():
    t0 = time.time()
    art = joblib.load(HERE / "final_preprocessor.joblib")
    df = pd.read_parquet(HERE / "final_train_features.parquet")
    feats = list(art.feature_columns)
    assert feats == [c for c in df.columns if c not in ("id", "fs_year", "label")], "feature order mismatch"
    y = df.label.to_numpy()
    rate = float(y.mean())
    assert abs(rate - CAL["training_rate_assumed"]) < 1e-12, (rate, CAL["training_rate_assumed"])
    booster = lgb.train(PARAMS, lgb.Dataset(df[feats], y, params={"max_bin": 255, "feature_pre_filter": False,
                                                                   "bin_construct_sample_cnt": 10**7}),
                        num_boost_round=N_TREES)
    booster.save_model(str(HERE / "final_lightgbm.txt"), num_iteration=N_TREES)
    p_raw = booster.predict(df[feats], num_iteration=N_TREES)
    lo = logit(np.clip(p_raw, 1e-6, 1 - 1e-6))
    a = brentq(lambda a: expit(a + lo).mean() - rate, -3, 3)
    delta = float(logit(TAU) - logit(rate))
    pd_final = np.clip(expit(a + lo + delta), FLOOR, CAP)
    params = {
        "config": asdict(art.config),
        "schema_version": art.feature_schema_version,
        "schema_hash": art.feature_schema_hash,
        "feature_columns": feats,
        "model_feature_names": booster.feature_name(),
        "medians": json.loads(art.sector_medians.to_json(orient="split", double_precision=15)),
        "industry_mapping": {str(k): float(v) for k, v in art.industry_mapping.to_dict().items()},
        "industry_global_mean": float(art.industry_global_mean),
        "fallback_pd": TAU,
        "fallback_pd_source": "tau: frozen central tendency used for the level (D-072/D-150), 5-month label",
        "legal_categories": list(art.legal_categories),
        "city_categories": list(art.city_categories),
        "best_iteration": N_TREES,
        "calibration": {"method": "prior_shift", "epsilon": 1e-6, "anchor_a": float(a), "slope_b": 1.0,
                        "delta": delta, "tau": TAU, "training_rate": rate, "floor": FLOOR, "cap": CAP,
                        "formula": "clip(expit(anchor_a + slope_b*logit(clip(p_raw, eps, 1-eps)) + delta), floor, cap)"},
        "model": {"library": "lightgbm", "version": lgb.__version__, "params": PARAMS, "n_trees": N_TREES,
                  "monotone_constraints": "none (KN decision 2026-11-21, D-159)",
                  "training_rows": int(len(df)), "training_defaults": int(y.sum()), "fs_years": "2007-2011"},
    }
    (HERE / "final_params.json").write_text(json.dumps(params, indent=2, allow_nan=False) + "\n")
    summ = dict(rows=len(df), defaults=int(y.sum()), train_rate=rate, mean_raw_pd=float(p_raw.mean()),
                anchor_a=float(a), delta=delta, tau=TAU, mean_final_pd=float(pd_final.mean()),
                insample_auc=float(roc_auc_score(y, p_raw)), insample_logloss_raw=float(log_loss(y, p_raw)),
                min_pd=float(pd_final.min()), max_pd=float(pd_final.max()),
                share_at_floor=float((pd_final <= FLOOR).mean()), seconds=round(time.time() - t0, 1),
                note="in-sample numbers are for sanity only; out-of-sample evidence is the Phase-1 walk-forward (D-158)")
    (HERE / "final_insample_summary.json").write_text(json.dumps(summ, indent=2) + "\n")
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
