"""Reference calibration study on Chloe (Yuxin)'s monotone LightGBM, round 1 (calibration-specialist, 2026-11-08).

Design (hers, "temporal_firm_year"): fit on FY2007 (features_train), early stopping on FY2008
(features_validation), test FY2009 (features_test). Nothing is fitted on FY2009; FY2009 labels only score.

Variants scored on FY2009 (all maps monotone increasing in the raw PD):
  raw        : booster probability
  b_platt    : HER method. Platt a + b*logit(p) fitted on FY2008 raw PDs (rows that also drove early stopping)
  c_ours     : OUR recipe (D-053): Platt slope b from the held-out slice (FY2008, same slope as b_platt),
               intercept re-anchored so that the mean PD over the model's own training rows (FY2007, in-sample
               PDs) equals the FY2007 default rate r_w (her label). Level = FY2007 rate.
  d_ours_e   : c_ours + prior shift delta = logit(tau_e) - logit(r_w) (D-072 M1), tau_e = rule e (D-071):
               default arrivals in months 7-12 of the last complete cohort (FY2008, her window: Dec 2009-May 2010),
               annualised. Known at the FY2009 decision date (1 Jun 2010 in her design).
  d_ours_b   : same with tau = rule b (FY2008 rate)            [sensitivity]
  d_ours_rw  : same with tau = D-073 random-walk hedge, anchor FY2008 rate, h = 1, sigma from the one available
               logit change (FY2007->FY2008)                     [sensitivity; sigma rests on one observation]
  d_s6level  : c_ours shifted to rule e computed on OUR start-6 label (ct_targets.csv, test 2009) [for our label]
  iso_val    : isotonic on FY2008 raw PDs (her code's isotonic option)                 [reference, D-072 rejected]
  iso_val_e  : iso_val + prior shift to rule e                                         [reference]
Floor 0.03% / cap 99% (D-075) applied to every variant; unclipped log loss also kept.

Sensitivity "clean slice": FY2008 split by firm into halves A/B; refit with early stopping on A, Platt slope on B.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.common import metrics as M  # noqa: E402

HERE = Path(__file__).resolve().parent
P_ = "2026-11-08_"
FEAT = Path("/Users/kietnguyen/Downloads/non-aggregated feature data sets (3 rounds)/round_01")
MANIFEST = ROOT / "production_lightgbm_v1/training_manifest.json"
META = ("id", "fs_year", "pred_start_dt", "label")
EPS, FLOOR, CAP = M.EPS, 0.0003, 0.99
DATA_END = pd.Timestamp("2013-05-31")

# Chloe's MONOTONE_DIRECTIONS, copied verbatim from
# `git show teammate/project-update:src/finance_project/model_development/lightgbm_model.py` (commit cd5667b).
MONOTONE_DIRECTIONS = {
    "equity_ratio": -1, "debt_ratio": 1, "st_debt_share": 1, "lt_debt_to_capital": 1, "bank_debt_conc": 1,
    "cash_ratio": -1, "cash_to_debt": -1, "cash_to_financing_exp_ratio": -1, "cur_asst_ratio": -1, "AR_ratio": 1,
    "payables_ratio": 1, "wc_ratio": -1, "cur_debt_ratio": 1, "annualized_change_in_roa": -1, "roa": -1,
    "ebitda_ratio": -1, "op_margin": -1, "net_prof_margin": -1, "cogs_ratio": 1, "int_coverage": -1,
    "financing_expense_ratio": 1, "financing_income_ratio": -1, "operating_cash_flow_ratio": -1,
    "operating_cash_flow_to_debt": -1, "debt_to_ebitda": 1, "loss_flag": 1, "ateco_sector_te": 1,
}


def logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def expit(z):
    return 1 / (1 + np.exp(-np.asarray(z, float)))


def solve_shift(z, target):  # same bisection as s3_methods.py
    lo, hi = -10.0, 10.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if expit(z + mid).mean() > target:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def platt(p, y):  # her ProbabilityCalibrator('platt'): LR(C=1e6) on logit(clip(p, 1e-6))
    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000).fit(logit(p).reshape(-1, 1), np.asarray(y, int))
    return float(lr.intercept_[0]), float(lr.coef_[0, 0])


# ------------------------------------------------------------------ 1. data
man = json.load(open(MANIFEST))
cfg = man["lightgbm_config"]
S = {s: pd.read_csv(FEAT / f"features_{s}.csv", low_memory=False) for s in ("train", "validation", "test")}
feats = [c for c in S["train"].columns if c not in META]
assert feats == man["feature_columns"], "feature list differs from the production manifest"
checks = []
for s, f in S.items():
    checks.append({"file": f"features_{s}.csv", "rows": len(f), "fs_years": str(sorted(f.fs_year.unique().tolist())),
                   "pred_start_dt": str(sorted(f.pred_start_dt.unique().tolist())), "defaults": int(f.label.sum()),
                   "default_rate": f.label.mean(), "dup_id_fs_year": int(f.duplicated(["id", "fs_year"]).sum()),
                   "n_features": len(feats)})
X = {s: f.reindex(columns=feats).apply(pd.to_numeric, errors="coerce") for s, f in S.items()}
Y = {s: f.label.astype(int).to_numpy() for s, f in S.items()}

# ------------------------------------------------------------------ 2. fit her LightGBM (her fit function, verbatim settings)
from lightgbm import LGBMClassifier, early_stopping, log_evaluation  # noqa: E402
import lightgbm  # noqa: E402

cons = [MONOTONE_DIRECTIONS.get(c, 0) for c in feats]
manifest_cons = man["model_details"]["monotone_constraints"]
assert all(manifest_cons[c] == MONOTONE_DIRECTIONS.get(c, 0) for c in feats), "constraints differ from manifest"


def fit(Xtr, ytr, Xes, yes):
    est = LGBMClassifier(objective="binary", n_estimators=5000, learning_rate=cfg["learning_rate"],
                         num_leaves=cfg["num_leaves"], min_child_samples=cfg["min_child_samples"],
                         subsample=cfg["subsample"], colsample_bytree=cfg["colsample_bytree"],
                         reg_lambda=cfg["reg_lambda"], monotone_constraints=cons,
                         random_state=cfg["random_state"], n_jobs=-1, verbosity=-1)
    est.fit(Xtr, ytr, eval_set=[(Xes, yes)], eval_metric=["auc", "average_precision", "binary_logloss"],
            callbacks=[early_stopping(cfg["early_stopping_rounds"], verbose=False), log_evaluation(0)])
    return est


model = fit(X["train"], Y["train"], X["validation"], Y["validation"])
raw = {s: model.predict_proba(X[s])[:, 1] for s in X}   # sklearn API predicts at best_iteration_
fit_info = {"lightgbm_version": lightgbm.__version__, "best_iteration": int(model.best_iteration_),
            "n_constrained": int(np.sum(np.array(cons) != 0)), "n_features": len(feats),
            "test_auc_raw_her_label": M.auc(Y["test"], raw["test"]), "her_reported_test_auc": 0.8431}
fit_info["auc_diff_vs_reported"] = fit_info["test_auc_raw_her_label"] - 0.8431

# ------------------------------------------------------------------ 3. labels: her label vs our panel
pan = pd.read_parquet(ROOT / "data/interim/panel.parquet",
                      columns=["row_id", "id", "fs_year", "stmt_date", "def_date", "y_s6", "status_s6"])
pan["id"], pan["fs_year"] = pan.id.astype("int64"), pan.fs_year.astype("int64")
# rebuild her label from the panel: def_date in [1 Jun t+1, 31 May t+2]; rows defaulted before 1 Jun t+1 dropped
ws = pd.to_datetime((pan.fs_year + 1).astype(str) + "-06-01")
we = pd.to_datetime((pan.fs_year + 2).astype(str) + "-05-31")
pan["alive_c"] = pan.def_date.isna() | (pan.def_date >= ws)
pan["y_c"] = ((pan.def_date >= ws) & (pan.def_date <= we)).astype(int)
pan["complete_c"] = we <= DATA_END
join_rows = []
for s in ("train", "validation", "test"):
    f = S[s][["id", "fs_year", "label"]].merge(pan, on=["id", "fs_year"], how="left", indicator=True)
    rebuilt_match = int((f.y_c == f.label).sum())
    st = f.status_s6.astype(str).where(f._merge == "both", "not in panel")
    for k, v in st.value_counts().items():
        join_rows.append({"file": s, "panel_status_s6": k, "rows": int(v), "her_label_defaults": int(f.label[st == k].sum())})
    join_rows.append({"file": s, "panel_status_s6": "CHECK her label rebuilt from panel matches", "rows": rebuilt_match,
                      "her_label_defaults": len(f)})
    # reverse: our 'ok' rows of that year absent from her file
    yr = int(S[s].fs_year.iloc[0])
    ours = pan[(pan.fs_year == yr) & (pan.status_s6 == "ok")]
    miss = ours.merge(S[s][["id", "fs_year"]], on=["id", "fs_year"], how="left", indicator=True).query("_merge=='left_only'")
    join_rows.append({"file": s, "panel_status_s6": "REVERSE: our ok rows missing from her file", "rows": len(miss),
                      "her_label_defaults": int(miss.y_s6.sum())})
pd.DataFrame(join_rows).to_csv(HERE / f"{P_}label_join.csv", index=False)
pd.DataFrame(checks).to_csv(HERE / f"{P_}file_checks.csv", index=False)

# ------------------------------------------------------------------ 4. cohort rates and level targets (her window)
def cohort_rate(t):
    c = pan[(pan.fs_year == t) & pan.alive_c]
    return len(c), int(c.y_c.sum()), c.y_c.mean()


def half_year_e(t):  # s2_rate_series.py rule e, on her window: months 7-12 of cohort t, annualised
    c = pan[(pan.fs_year == t) & pan.alive_c]
    w0 = pd.Timestamp(f"{t + 1}-06-01")
    h = []
    for j in range(6, 12):
        m0, m1 = w0 + pd.DateOffset(months=j), w0 + pd.DateOffset(months=j + 1)
        at = (c.def_date.isna() | (c.def_date >= m0)).sum()
        h.append(((c.def_date >= m0) & (c.def_date < m1)).sum() / at)
    return float(1 - np.prod(1 - np.array(h))) * 2


rates = pd.DataFrame([dict(zip(["fs_year", "n", "defaults", "rate"], (t, *cohort_rate(t)))) for t in range(2007, 2012)])
rates["half_year_e_months7_12"] = [half_year_e(t) for t in range(2007, 2012)]
rates.to_csv(HERE / f"{P_}rates_by_cohort_her_window.csv", index=False)
R = rates.set_index("fs_year")
r_w = float(Y["train"].mean())                       # FY2007, the model's own fitting rows
rng = np.random.default_rng(0)
eps = rng.standard_normal(400_000)                   # as s5_final_level.py
sig_dec = float(abs(logit(R.rate[2008]) - logit(R.rate[2007])))
targets = {"e_last_6m_annualised": float(R.half_year_e_months7_12[2008]), "b_last_year": float(R.rate[2008]),
           "a_pooled_2007_2008": float((R.defaults[2007] + R.defaults[2008]) / (R.n[2007] + R.n[2008])),
           "rw_hedge_h1_anchor_b": float(expit(logit(R.rate[2008]) + sig_dec * eps).mean())}
ct6 = pd.read_csv(ROOT / "reports/calibration/ct_targets.csv").query("label=='s6' and test_year==2009")
tau_s6_e = float(ct6.query("rule=='e_last_6m_annualised'").target.iloc[0])
pd.DataFrame([{"rule": k, "tau": v, "r_w_fy2007": r_w, "delta": logit(v) - logit(r_w),
               "observed_fy2009_her_label": float(Y["test"].mean())} for k, v in targets.items()] +
             [{"rule": "e on OUR s6 label (ct_targets.csv)", "tau": tau_s6_e, "r_w_fy2007": r_w,
               "delta": logit(tau_s6_e) - logit(r_w), "observed_fy2009_her_label": np.nan}]
             ).to_csv(HERE / f"{P_}level_targets_fy2009.csv", index=False)

# ------------------------------------------------------------------ 5. variants
a_v, b_v = platt(raw["validation"], Y["validation"])                       # her Platt
z_te, z_tr = logit(raw["test"]), logit(raw["train"])
a_anchor = solve_shift(b_v * z_tr, r_w)                                    # D-053 anchor on FY2007 in-sample
iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(raw["validation"], Y["validation"])
p_iso = iso.predict(raw["test"])
p_iso_tr = iso.predict(raw["train"])
a_iso = solve_shift(logit(p_iso_tr), r_w)
d = {k: logit(v) - logit(r_w) for k, v in targets.items()}
V = {"raw": raw["test"],
     "b_platt": expit(a_v + b_v * z_te),
     "c_ours": expit(a_anchor + b_v * z_te),
     "d_ours_e": expit(a_anchor + b_v * z_te + d["e_last_6m_annualised"]),
     "d_ours_b": expit(a_anchor + b_v * z_te + d["b_last_year"]),
     "d_ours_rw": expit(a_anchor + b_v * z_te + d["rw_hedge_h1_anchor_b"]),
     "d_s6level": expit(a_anchor + b_v * z_te + logit(tau_s6_e) - logit(r_w)),
     "iso_val": p_iso,
     "iso_val_e": expit(logit(p_iso) + a_iso + d["e_last_6m_annualised"])}
params = {"best_iteration": fit_info["best_iteration"], "platt_a_fy2008": a_v, "platt_b_fy2008": b_v,
          "anchor_a_fy2007": a_anchor, "r_w_fy2007": r_w, "mean_raw_pd_fy2007_insample": float(raw["train"].mean()),
          "mean_raw_pd_fy2008": float(raw["validation"].mean()), "iso_anchor": a_iso, **{f"delta_{k}": v for k, v in d.items()},
          "tau_s6_e": tau_s6_e, "iso_n_steps": int(len(iso.X_thresholds_))}

# ------------------------------------------------------------------ 6. clean-slice sensitivity (FY2008 halves by firm)
ids8 = S["validation"].id.to_numpy()
u = np.unique(ids8)
half_b = np.isin(ids8, np.random.default_rng(0).choice(u, len(u) // 2, replace=False))
m2 = fit(X["train"], Y["train"], X["validation"][~half_b], Y["validation"][~half_b])
r2 = {s: m2.predict_proba(X[s])[:, 1] for s in X}
a2, b2 = platt(r2["validation"][half_b], Y["validation"][half_b])
anc2 = solve_shift(b2 * logit(r2["train"]), r_w)
V["c_ours_cleanB"] = expit(anc2 + b2 * logit(r2["test"]))
V["d_ours_e_cleanB"] = expit(anc2 + b2 * logit(r2["test"]) + d["e_last_6m_annualised"])
params.update({"cleanB_best_iteration": int(m2.best_iteration_), "cleanB_platt_a": a2, "cleanB_platt_b": b2,
               "cleanB_anchor": anc2, "cleanB_rows_B": int(half_b.sum()), "cleanB_defaults_B": int(Y["validation"][half_b].sum())})
json.dump({**fit_info, **params}, open(HERE / f"{P_}fit_and_map_params.json", "w"), indent=1)

# ------------------------------------------------------------------ 7. predictions file (contract-like)
te = S["test"][["id", "fs_year", "label"]].rename(columns={"label": "y_chloe"}).copy()
te = te.merge(pan[["id", "fs_year", "row_id", "y_s6", "status_s6"]], on=["id", "fs_year"], how="left")
te["row_id"] = te.id.astype("int64") * 10000 + te.fs_year.astype("int64")
te["fold"] = "chloe_r1_fy2009"
for k, v in V.items():
    te[f"pd_{k}"] = np.clip(v, FLOOR, CAP)
    te[f"pdu_{k}"] = v
te.to_parquet(HERE / f"{P_}fy2009_predictions.parquet", index=False)
va = S["validation"][["id", "fs_year", "label"]].copy()
va["pd_raw"] = raw["validation"]
va.to_parquet(HERE / f"{P_}fy2008_raw_predictions.parquet", index=False)

# ------------------------------------------------------------------ 8. metrics per variant x label
rows, dec = [], []
labs = {"her_label": (np.ones(len(te), bool), te.y_chloe.to_numpy()),
        "our_label_s6": ((te.status_s6 == "ok").fillna(False).to_numpy(), te.y_s6.to_numpy())}
for lab, (mk, yy) in labs.items():
    y = yy[mk].astype(int)
    for k in V:
        p, pu = te[f"pd_{k}"].to_numpy()[mk], te[f"pdu_{k}"].to_numpy()[mk]
        r = {"label": lab, "variant": k, "n": int(mk.sum()), "defaults": int(y.sum()), "auc": M.auc(y, p),
             "auc_unclipped": M.auc(y, pu), "log_loss": M.log_loss(y, p), "log_loss_unclipped": M.log_loss(y, pu),
             "brier": M.brier(y, p), "n_at_floor": int((pu < FLOOR).sum()), "n_at_cap": int((pu > CAP).sum())}
        r.update(M.calibration_in_the_large(y, p))
        r.update(M.calibration_slope_intercept(y, p))
        rows.append(r)
        t = M.decile_table(y, p, 10)
        t.insert(0, "variant", k)
        t.insert(0, "label", lab)
        dec.append(t)
res = pd.DataFrame(rows)
res["auc_minus_raw"] = res.auc_unclipped - res.groupby("label").auc_unclipped.transform(lambda s: s.iloc[0])
res.to_csv(HERE / f"{P_}metrics_fy2009.csv", index=False)
pd.concat(dec).to_csv(HERE / f"{P_}deciles_fy2009.csv", index=False)

# ------------------------------------------------------------------ 9. production level parameters (her window, D-073 logic)
z_c = logit(R.rate.to_numpy())                         # FY2007..FY2011, all complete under her window
sigma = float(np.sqrt(np.mean(np.diff(z_c) ** 2)))     # driftless logit random walk, as s5_final_level.py
anchor11 = float(R.rate[2011])
rw = {h: float(expit(logit(anchor11) + sigma * np.sqrt(h) * eps).mean()) for h in (1, 2, 3)}
pooled = float(R.defaults.sum() / R.n.sum())
r_prod = float(man["training_default_rate"])
cand = pd.DataFrame([
    {"candidate": "a pooled FY2007-2011 (her window)", "tau": pooled},
    {"candidate": "b last cohort FY2011", "tau": anchor11},
    {"candidate": "e months 7-12 of FY2011 cohort (Dec 2012-May 2013) annualised", "tau": float(R.half_year_e_months7_12[2011])},
    {"candidate": "d equal-weight FY2007-2011", "tau": float(R.rate.mean())},
    *[{"candidate": f"RW hedge anchor FY2011, h={h}", "tau": v} for h, v in rw.items()]])
cand["delta_vs_production_rate"] = logit(cand.tau) - logit(r_prod)
cand.to_csv(HERE / f"{P_}production_level_candidates.csv", index=False)
TAU = rw[2]
prod = {
    "created_by": "calibration-specialist, experiments/calibration/2026-11-08_chloe_lgbm_v1/2026-11-08_r1_fit_and_calibrate.py",
    "model": "production_lightgbm_v1 (Chloe/Yuxin), trained FY2007-2011, 835,016 rows",
    "status": "REFERENCE ONLY - not wired into the harness; see the decision log entry of 2026-11-08",
    "label_window": "Chloe's: default in [1 Jun fs_year+1, 31 May fs_year+2]; rows defaulted before 1 Jun fs_year+1 excluded",
    "pipeline": "PD = clip(expit(anchor_a + slope_b*logit(p_raw) + delta), floor, cap)",
    "slope_b": b_v, "slope_b_source": "round-1 Platt slope on FY2008 (model fit on FY2007); a production-specific slope needs an "
               "out-of-time slice for the 2007-2011 model, which does not exist; clean-slice check slope in fit_and_map_params.json",
    "anchor_a": None,
    "anchor_a_procedure": "solve anchor_a so that mean(expit(anchor_a + slope_b*logit(p_raw))) over the 835,016 training rows "
                          "equals training_rate_assumed (bisection, as solve_shift in s3_methods.py). Needs the production model's "
                          "in-sample raw PDs on its training rows, which were not handed over; compute once at estimation time.",
    "training_rate_assumed": r_prod, "training_rate_recomputed_from_panel_her_window": pooled,
    "tau": TAU, "tau_rule": "D-073 logic: expected rate h=2 years after the FY2011 anchor under a driftless logit random walk",
    "tau_anchor_fy2011": anchor11, "sigma_logit_per_year": sigma, "tau_by_horizon": rw,
    "delta": float(logit(TAU) - logit(r_prod)),
    "floor": FLOOR, "cap": CAP,
    "replaces": "her Platt calibrator fitted on pooled round 1-3 out-of-time raw PDs (default rate 1.536%, mean raw PD 1.105%, "
                "manifest calibration_raw_metrics); applying that intercept to a model refit on those same years double-counts "
                "the level (D-053 alternative b)",
}
json.dump(prod, open(HERE / f"{P_}params_for_production.json", "w"), indent=1)

print(json.dumps({**fit_info, **params}, indent=1))
print(pd.DataFrame(checks).to_string()); print(pd.DataFrame(join_rows).to_string()); print(rates.to_string())
print(cand.to_string())
print(res[["label", "variant", "auc", "auc_minus_raw", "log_loss", "brier", "mean_pd", "observed_dr", "ratio_obs_to_pred",
           "cal_slope", "cal_intercept"]].round(5).to_string())
