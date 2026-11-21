"""Team Turquoise: estimation code for the final 1-year PD model (mono20_nogw).

Starts from Roger's raw training file and produces the two frozen files the prediction code reads:
    lightgbm.txt        the trained LightGBM trees
    preprocessing.json  every other parameter: imputation medians, industry target encoding, category lists,
                        ordered model features, calibration constants, fallback PD, model settings

Consolidates 01 (final) -> 02 -> 06 -> 10 and the tau rule of
experiments/calibration/2026-11-08_chloe_lgbm_v1/2026-11-08_r1_fit_and_calibrate.py into one script whose only
input is the raw file. Shipped to teammates as estimation/estimate.py in the full-codebase package.
Usage (repo root, estimation env = the dev versions: lightgbm 4.7.0, pandas 2.3.3, numpy 2.3.5, sklearn 1.7.2):
    python experiments/final/2026-11-21_final_fit/2026-11-21_11_estimate_end_to_end.py --raw data/raw/train-2025.csv
    python experiments/final/2026-11-21_final_fit/2026-11-21_12_verify_rebuild.py
Default output: rebuild/ in this folder (git-ignored; the frozen copy lives in harness/2026-11-21_variants/mono20_nogw).
Runtime about 3-4 minutes on a laptop (the team preprocessing is the slow part), peak memory a few GB.

Steps (decision-log IDs in brackets; docs/DECISION_LOG.md in the team repo):
  1. Label [D-151]. For a statement of fiscal year t: y = 1 if def_date is in [1 Jun t+1, 1 Jun t+2).
     Rows with a default before 1 Jun t+1 (already defaulted when the statement is filed) are dropped. Only
     FY2007-2011 have a complete outcome window in the data. Expected: 835,016 rows, 11,336 defaults.
     def_date is used ONLY here; it is dropped before any feature is built.
  2. Preprocessing and features [D-156]. The team code (Chloe and Desmond's schema 1.3.0, vendored unchanged
     in vendor_7949207/, branch teammate/project-update @ 7949207): anomaly flags, missing-value rules,
     industry medians, ratios, out-of-fold industry target encoding, legal-form and region dummies, and the
     75 VIF-selected features. All of it is fitted on the training rows only.
  3. Feature set [D-164]. The 75 VIF features minus goodwill_ratio (the raw field behaves like value added,
     not accounting goodwill, D-010/D-091): 74 model features.
  4. Monotone constraints [D-157]. The expected-sign table (experiments/gbm/2026-11-20_grid75/sign_table.csv, column `constraint`):
     20 features constrained, the rest free. Method "intermediate".
  5. LightGBM [D-158]. Grid winner from the walk-forward: num_leaves 31, min_child_samples 3000,
     feature_fraction 0.5, lambda_l2 10, learning_rate 0.05, no bagging, deterministic. 230 trees = mean
     round-3 best iteration of this configuration (243, 216); there is no validation set left for early
     stopping once all years are used.
  6. Level calibration [D-072, D-073, D-075]. Prior shift:
         PD = clip( expit(a + logit(p_raw) + delta), 0.0003, 0.99 )
     a     : solved so the mean in-sample PD equals the training default rate (removes the raw model's bias)
     tau   : central tendency for the holdout = expected default rate two years after the FY2011 cohort under
             a driftless random walk in logit(rate); sigma = root-mean-square yearly logit change FY2007-2011;
             expectation by Monte Carlo (400,000 normal draws, seed 0), exactly as in the calibration study
     delta : logit(tau) - logit(training default rate)
     Floor 0.03% and cap 99% [D-075]. tau is also the fallback PD for rows whose features cannot be computed.
No holdout data, no test data and no LLM are used anywhere.
"""
import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit, logit

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "vendor_7949207" / "src"))
from finance_project.data_processing.preprocess import PreprocessingConfig, fit_preprocessor  # noqa: E402

YEARS = range(2007, 2012)
DROP = "goodwill_ratio"
N_TREES = 230
FLOOR, CAP, EPS = 0.0003, 0.99, 1e-6
TAU_HORIZON_YEARS, TAU_DRAWS, TAU_SEED = 2, 400_000, 0
LGB_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_child_samples=3000,
                  feature_fraction=0.5, bagging_fraction=1.0, bagging_freq=0, lambda_l2=10.0, max_bin=255,
                  bin_construct_sample_cnt=10**7, feature_pre_filter=False, seed=42, deterministic=True,
                  force_row_wise=True, num_threads=8, verbose=-1)


def read_raw(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def build_labels(raw: pd.DataFrame) -> pd.DataFrame:
    """Step 1 (D-151): keep labelled FY2007-2011 rows, add `label`, drop def_date."""
    t = raw["fs_year"]
    d = pd.to_datetime(raw["def_date"], errors="coerce")
    win_start = pd.to_datetime((t + 1).astype(str) + "-06-01")
    win_end = pd.to_datetime((t + 2).astype(str) + "-06-01")
    already_defaulted = d.notna() & (d < win_start)
    keep = t.isin(YEARS) & ~already_defaulted
    out = raw.loc[keep].copy()
    out["label"] = (d[keep].notna() & (d[keep] >= win_start[keep]) & (d[keep] < win_end[keep])).astype(int)
    return out.drop(columns=["def_date"]).reset_index(drop=True)


def central_tendency(lab: pd.DataFrame) -> dict:
    """Step 6: tau from the cohort default rates (D-073 logic, 5-month label)."""
    rates = lab.groupby("fs_year")["label"].agg(n="size", defaults="sum")
    rates["rate"] = rates.defaults / rates.n
    z = logit(rates.rate.to_numpy())
    sigma = float(np.sqrt(np.mean(np.diff(z) ** 2)))           # driftless logit random walk
    anchor = float(rates.rate.loc[max(YEARS)])
    draws = np.random.default_rng(TAU_SEED).standard_normal(TAU_DRAWS)
    tau = float(expit(logit(anchor) + sigma * np.sqrt(TAU_HORIZON_YEARS) * draws).mean())
    return dict(tau=tau, anchor_fy2011=anchor, sigma_logit_per_year=sigma,
                rates_by_fs_year={int(k): float(v) for k, v in rates.rate.items()})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, required=True, help="train-2025.csv (or its parquet copy)")
    ap.add_argument("--out", type=Path, default=HERE / "rebuild")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # 1. labels
    lab = build_labels(read_raw(a.raw))
    rate = float(lab.label.mean())
    print(f"[1] labelled rows {len(lab):,} (expected 835,016), defaults {int(lab.label.sum()):,} (expected 11,336), "
          f"rate {rate:.6%}", flush=True)

    # 2. team preprocessing, fitted on the training rows only
    feats_df, art = fit_preprocessor(lab, lab, PreprocessingConfig(feature_mode="firm_year"))
    print(f"[2] preprocessing fitted, {len(art.feature_columns)} VIF features ({time.time() - t0:.0f}s)", flush=True)

    # 3. model features
    feats = [f for f in art.feature_columns if f != DROP]
    assert len(feats) == 74

    # 4. monotone constraints from the sign table
    sign = pd.read_csv(HERE.parents[2] / "experiments/gbm/2026-11-20_grid75/sign_table.csv").set_index("feature")["constraint"]
    mono = [int(sign[f]) for f in feats]
    assert sum(1 for c in mono if c) == 20
    params = dict(LGB_PARAMS, monotone_constraints=mono, monotone_constraints_method="intermediate")

    # 5. LightGBM
    y = feats_df["label"].to_numpy()
    ds = lgb.Dataset(feats_df[feats], y, params={"max_bin": 255, "feature_pre_filter": False,
                                                 "bin_construct_sample_cnt": 10**7})
    booster = lgb.train(params, ds, num_boost_round=N_TREES)
    booster.save_model(str(a.out / "lightgbm.txt"), num_iteration=N_TREES)
    print(f"[5] LightGBM trained, {N_TREES} trees ({time.time() - t0:.0f}s)", flush=True)

    # 6. level calibration
    p_raw = booster.predict(feats_df[feats], num_iteration=N_TREES)
    lo = logit(np.clip(p_raw, EPS, 1 - EPS))
    anchor_a = float(brentq(lambda s: expit(s + lo).mean() - rate, -3, 3))
    ct = central_tendency(lab)
    tau = ct["tau"]
    delta = float(logit(tau) - logit(rate))
    pd_final = np.clip(expit(anchor_a + lo + delta), FLOOR, CAP)
    print(f"[6] tau {tau:.6%}, delta {delta:.6f}, anchor a {anchor_a:.7f}", flush=True)

    # frozen parameter file, same structure as harness/artifacts/preprocessing.json
    out = {
        "config": asdict(art.config),
        "schema_version": art.feature_schema_version,
        "schema_hash": art.feature_schema_hash,
        "feature_columns": feats,
        "model_feature_names": booster.feature_name(),
        "medians": json.loads(art.sector_medians.to_json(orient="split", double_precision=15)),
        "industry_mapping": {str(k): float(v) for k, v in art.industry_mapping.to_dict().items()},
        "industry_global_mean": float(art.industry_global_mean),
        "fallback_pd": tau,
        "fallback_pd_source": "tau: frozen central tendency used for the level (D-072/D-150), 5-month label",
        "legal_categories": list(art.legal_categories),
        "city_categories": list(art.city_categories),
        "best_iteration": N_TREES,
        "calibration": {"method": "prior_shift", "epsilon": EPS, "anchor_a": anchor_a, "slope_b": 1.0,
                        "delta": delta, "tau": tau, "training_rate": rate, "floor": FLOOR, "cap": CAP,
                        "formula": "clip(expit(anchor_a + slope_b*logit(clip(p_raw, eps, 1-eps)) + delta), floor, cap)"},
        "model": {"library": "lightgbm", "version": lgb.__version__, "params": params, "n_trees": N_TREES,
                  "monotone_constraints": {f: c for f, c in zip(feats, mono) if c != 0},
                  "training_rows": int(len(feats_df)), "training_defaults": int(y.sum()), "fs_years": "2007-2011",
                  "variant": "mono20_nogw",
                  "description": "full monotone (20 constraints, D-157) without goodwill_ratio, 74 features (D-164)",
                  "dropped_features": [DROP]},
    }
    (a.out / "preprocessing.json").write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")
    summary = dict(rows=len(feats_df), defaults=int(y.sum()), train_rate=rate, **ct, delta=delta, anchor_a=anchor_a,
                   mean_raw_pd=float(p_raw.mean()), mean_final_pd=float(pd_final.mean()),
                   min_pd=float(pd_final.min()), max_pd=float(pd_final.max()), seconds=round(time.time() - t0, 1),
                   note="in-sample numbers are a sanity check only; out-of-sample evidence is the walk-forward")
    (a.out / "estimation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {a.out / 'lightgbm.txt'}, {a.out / 'preprocessing.json'}, estimation_summary.json "
          f"({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
