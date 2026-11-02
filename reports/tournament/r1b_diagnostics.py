"""Referee round 1b diagnostics (referee, 2026-11-02).

    OMP_NUM_THREADS=2 /Users/kietnguyen/miniconda3/envs/datascience/bin/python reports/tournament/r1b_diagnostics.py

LOCKBOX: reads the panel with a parquet filter fs_year < 2011; never opens files holding 2011/2012 rows.
All metrics from src/common/metrics.py; the firm-clustered paired bootstrap is the fast estimator of
score_r1.py (AUCEngine / bootstrap_draws, verified equal to metrics.paired_bootstrap: r1b_bootstrap_equivalence.csv).

Tasks:  1 GAM audit (contract, reproduction w/o label columns, face validity, metrics, paired bootstrap)
        2 power by segment (year, size quintile with training-fold edges, ATECO section, seen/new)
        3 missing-input calibration check (lesson L1)
        4 grade grid with <= 12% of firms per grade, edges on training-year predictions; 2010 table
        5 overfitting / validation evidence
Writes reports/tournament/r1b_*.csv|json and fig/r1b_*.png.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "2")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments/gam"))
from src.common import metrics as M  # noqa: E402
from src.features import build as B  # noqa: E402
import gam_lib as GL  # noqa: E402

OUT = ROOT / "reports/tournament"
FIG = OUT / "fig"
FIG.mkdir(parents=True, exist_ok=True)
N_BOOT, SEED = int(os.environ.get("R1B_NBOOT", 1000)), 0
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def logit(p):
    p = np.clip(np.asarray(p, float), M.EPS, 1 - M.EPS)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1 / (1 + np.exp(-np.asarray(z, float)))


def cp_ci(k, n):
    if n == 0:
        return np.nan, np.nan
    c = stats.binomtest(int(k), int(n)).proportion_ci(confidence_level=0.95, method="exact")
    return c.low, c.high


# ------------------------------------------------------------------------------------------------
# 0. Data (LOCKBOX filter) and predictions
# ------------------------------------------------------------------------------------------------
splits = json.load(open(ROOT / "data/splits/walkforward.json"))
testrows = pd.read_parquet(ROOT / "data/splits/walkforward_test_rows.parquet")
assert testrows["test_year"].max() <= 2010
panel = pd.read_parquet(ROOT / "data/interim/panel.parquet", filters=[("fs_year", "<", 2011)])
assert panel["fs_year"].max() <= 2010
LABEL_COLS = [c for c in panel.columns if c in ("def_date", "y", "label_status", "win_start", "win_end", "src_index")
              or c.startswith("y_") or c.startswith("status_")]
base = testrows.merge(panel, on="row_id", how="left", validate="one_to_one")
assert (base["fs_year"] == base["test_year"]).all() and base["y"].notna().all()
y = base["y"].astype(int).to_numpy()
firm = base["id"].to_numpy()
year = base["test_year"].to_numpy()
seen = base["firm_seen_in_train"].to_numpy(bool)
fold = base["fold"].to_numpy()
log("rows", len(base), "defaults", y.sum())

FILES = {
    "GB-01": "experiments/gbm/GB-01/oot_predictions.parquet",
    "CAL-01": "experiments/calibration/CAL-01/oot_predictions.parquet",
    "SC-03": "experiments/scorecard/SC-03/oot_predictions.parquet",
    "RC-02": "experiments/riskcalc/RC-02/oot_predictions.parquet",
    "GAM-01": "experiments/gam/GAM-01/oot_predictions.parquet",
    "GAM-01+CAL": "experiments/gam/GAM-01/oot_predictions_calibrated.parquet",
    "GAM-01raw": "experiments/gam/GAM-01/oot_predictions_raw.parquet",
    "GAM-02": "experiments/gam/GAM-02/oot_predictions.parquet",
    "GAM-02+CAL": "experiments/gam/GAM-02/oot_predictions_calibrated.parquet",
    "GAM-02raw": "experiments/gam/GAM-02/oot_predictions_raw.parquet",
}
P, contract = {}, []
for e, f in FILES.items():
    o = pd.read_parquet(ROOT / f)
    m = testrows[["row_id", "fold"]].merge(o, on=["row_id", "fold"], how="left", indicator=True)
    r = {"exp": e, "file": f, "n_rows": len(o), "expected_rows": len(testrows),
         "dup_row_fold": int(o.duplicated(["row_id", "fold"]).sum()), "nan_pd": int(o["pd"].isna().sum()),
         "pd_outside_open_01": int((~o["pd"].between(0, 1, inclusive="neither")).sum()),
         "missing_test_rows": int((m["_merge"] == "left_only").sum()),
         "min_pd": float(o["pd"].min()), "max_pd": float(o["pd"].max())}
    r["extra_rows"] = len(o) - int((m["_merge"] == "both").sum())
    r["PASS"] = (r["n_rows"] == r["expected_rows"] and r["dup_row_fold"] == 0 and r["nan_pd"] == 0
                 and r["pd_outside_open_01"] == 0 and r["missing_test_rows"] == 0 and r["extra_rows"] == 0)
    contract.append(r)
    P[e] = base[["row_id", "fold"]].merge(o, on=["row_id", "fold"], how="left")["pd"].to_numpy(float)
pd.DataFrame(contract).to_csv(OUT / "r1b_contract.csv", index=False)
log(pd.DataFrame(contract)[["exp", "n_rows", "missing_test_rows", "nan_pd", "pd_outside_open_01", "PASS"]].to_string())

# ------------------------------------------------------------------------------------------------
# 1a. GAM reproduction from the saved fold models on test rows with every label column removed,
#     rows shuffled (row-wise prediction, no label access, saved artefacts = submitted PDs)
# ------------------------------------------------------------------------------------------------
repro, MODELS = [], {}
for e in ("GAM-01", "GAM-02"):
    cfg = json.load(open(ROOT / f"experiments/gam/{e}/config.json"))
    for fc in cfg["folds"]:
        k = fc["fold"]
        art = GL.load_model(str(ROOT / f"experiments/gam/{e}/model_fold{k}"))
        art["ebm"].n_jobs = 2
        MODELS[(e, k)] = art
        m = fold == k
        te = base.loc[m].drop(columns=LABEL_COLS + ["fold", "test_year", "firm_seen_in_train"])
        perm = np.random.default_rng(k).permutation(len(te))
        te_s = te.iloc[perm]
        t0 = time.time()
        p_own = np.empty(len(te)); p_own[perm] = GL.predict_pd(art, te_s)
        secs = time.time() - t0
        p_raw = np.empty(len(te)); p_raw[perm] = GL.predict_pd(art, te_s, raw=True)
        p_cal = np.empty(len(te)); p_cal[perm] = GL.predict_pd(art, te_s, delta=fc["delta_prior_shift"])
        repro.append({"exp": e, "fold": k, "n_test": int(m.sum()),
                      "max_abs_diff_own": float(np.abs(p_own - P[e][m]).max()),
                      "max_abs_diff_raw": float(np.abs(p_raw - P[e + "raw"][m]).max()),
                      "max_abs_diff_cal": float(np.abs(p_cal - P[e + "+CAL"][m]).max()),
                      "r_w_config": fc["r_w"], "tau_rule_e": fc["tau_rule_e"], "delta": fc["delta_prior_shift"],
                      "predict_secs_2threads": round(secs, 2), "chosen_cfg": json.dumps(fc["chosen_cfg"]),
                      "n_train_config": fc["n_train"]})
# r_w must equal the training default rate of the fold (training labels only)
for r in repro:
    f = [x for x in splits["folds"] if x["fold"] == r["fold"]][0]
    tr = panel[panel["fs_year"].isin(f["train_years"]) & (panel["label_status"] == "ok")]
    r["r_w_recomputed"] = float(tr["y"].mean())
    r["n_train_recomputed"] = len(tr)
repro = pd.DataFrame(repro)
repro.to_csv(OUT / "r1b_gam_repro.csv", index=False)
log(repro.to_string())

# ------------------------------------------------------------------------------------------------
# 1b. Face validity of the EBM shape functions (every fold; main terms; bins excl. missing/unknown)
# ------------------------------------------------------------------------------------------------
SIGNS = B.expected_signs()
fvr = []
for (e, k), art in MODELS.items():
    ebm = art["ebm"]
    for ti, tf in enumerate(ebm.term_features_):
        name = ebm.term_names_[ti]
        if len(tf) == 1:
            s = SIGNS.get(name, 0)
            sc = np.asarray(ebm.term_scores_[ti], float)
            body = sc[1:-1]
            d = np.diff(body) * (s if s != 0 else 1)
            tot = np.abs(d).sum()
            against = float(np.clip(-d, 0, None).sum() / tot) if (tot > 1e-12 and s != 0) else 0.0
            fvr.append({"exp": e, "fold": k, "term": name, "expected_sign": s, "n_bins": len(body),
                        "range_logodds": float(np.ptp(body)), "lowest_bin_score": float(body[0]),
                        "highest_bin_score": float(body[-1]), "missing_bin_score": float(sc[0]),
                        "against_share": against, "violates": bool(s != 0 and against > 1e-9)})
        else:
            names = [ebm.feature_names_in_[i] for i in tf]
            fvr.append({"exp": e, "fold": k, "term": name, "expected_sign": 0, "n_bins": -1,
                        "range_logodds": float(np.ptp(ebm.term_scores_[ti])),
                        "pair_involves_signed_feature": any(SIGNS.get(n, 0) != 0 for n in names),
                        "violates": False})
fvr = pd.DataFrame(fvr)
fvr.to_csv(OUT / "r1b_gam_face_validity.csv", index=False)
log("face-validity violations:", fvr.groupby(["exp", "fold"])["violates"].sum().to_dict())
log("pairs with signed features:", int(fvr.get("pair_involves_signed_feature", pd.Series(dtype=bool)).fillna(False).sum()))

# ------------------------------------------------------------------------------------------------
# 1c. Metrics (shared code): pooled, per year, seen/new
# ------------------------------------------------------------------------------------------------
grp = np.where(seen, "seen", "new")
MODS = ["GB-01", "CAL-01", "SC-03", "RC-02", "GAM-01", "GAM-01+CAL", "GAM-02", "GAM-02+CAL"]
rows = []
for e in MODS:
    s = pd.concat([M.score(y, P[e], year), M.score(y, P[e], grp).iloc[1:]])
    s["group"] = s["group"].astype(str)
    s.insert(0, "exp", e)
    rows.append(s)
met = pd.concat(rows, ignore_index=True)
met.to_csv(OUT / "r1b_metrics_long.csv", index=False)
dec = []
for e in ("GAM-01", "GAM-01+CAL", "GAM-02"):
    for t in ("ALL", 2008, 2009, 2010):
        mm = np.ones(len(y), bool) if t == "ALL" else year == t
        d = M.decile_table(y[mm], P[e][mm]); d.insert(0, "subset", t); d.insert(0, "exp", e); dec.append(d)
pd.concat(dec).to_csv(OUT / "r1b_gam_deciles.csv", index=False)

# ------------------------------------------------------------------------------------------------
# 2. Segment definitions
# ------------------------------------------------------------------------------------------------
# size quintiles: edges = quintiles of asst_tot on the fold's TRAINING rows (fs_year <= t, ok)
size_q = np.full(len(base), "missing", dtype=object)
size_edges = []
for f in splits["folds"]:
    tr = panel[panel["fs_year"].isin(f["train_years"]) & (panel["label_status"] == "ok")]
    ta = B.coerce_input(tr)["asst_tot"]
    ed = np.quantile(ta[ta > 0].dropna(), [0.2, 0.4, 0.6, 0.8])
    size_edges.append({"fold": f["fold"], **{f"q{i+1}_edge_eur": float(v) for i, v in enumerate(ed)}})
    m = fold == f["fold"]
    ta_te = B.coerce_input(base.loc[m])["asst_tot"].to_numpy(float)
    q = np.searchsorted(ed, ta_te, side="right") + 1
    lab = np.array([f"Q{i}" for i in q], dtype=object)
    lab[~np.isfinite(ta_te) | (ta_te <= 0)] = "missing"
    size_q[m] = lab
pd.DataFrame(size_edges).to_csv(OUT / "r1b_size_quintile_edges.csv", index=False)
# ATECO section (raw panel ateco_section); sections with < 30 test defaults pooled; L kept separately
sec = base["ateco_section"].astype("string").fillna("missing").to_numpy(object)
dsec = pd.Series(y).groupby(sec).sum()
small = [s for s, d in dsec.items() if d < 30 and s != "L"]
sec_cell = np.where(np.isin(sec, small), "pooled(" + ",".join(sorted(small)) + ")", sec).astype(object)

SEG = {"ALL": {"ALL": np.ones(len(y), bool)}}
SEG["year"] = {str(t): year == t for t in (2008, 2009, 2010)}
SEG["size"] = {q: size_q == q for q in sorted(pd.unique(size_q))}
SEG["ateco"] = {s: sec_cell == s for s in sorted(pd.unique(sec_cell))}
SEG["firm"] = {"seen": seen, "new": ~seen}
subsets = {f"{g}|{c}": m for g, d in SEG.items() for c, m in d.items() if m.sum() > 0}

# ------------------------------------------------------------------------------------------------
# Fast firm-clustered bootstrap (identical estimator to score_r1.py / metrics.paired_bootstrap)
# ------------------------------------------------------------------------------------------------
class AUCEngine:
    def __init__(self, yv, pv):
        o = np.argsort(pv, kind="mergesort")
        self.o, self.y = o, yv[o]
        ps = pv[o]
        self.starts = np.flatnonzero(np.r_[True, ps[1:] != ps[:-1]])

    def __call__(self, w):
        w = w[self.o]
        gp = np.add.reduceat(w * self.y, self.starts)
        gn = np.add.reduceat(w * (1 - self.y), self.starts)
        below = np.cumsum(gn) - gn
        return float((gp * (below + 0.5 * gn)).sum() / (gp.sum() * gn.sum()))


def row_loss(yv, pv):
    pv = np.clip(pv, M.EPS, 1 - M.EPS)
    return -(yv * np.log(pv) + (1 - yv) * np.log(1 - pv))


def bootstrap_draws(yv, firmv, preds, subs, n_boot=N_BOOT, seed=SEED, ll_subsets=None, auc_only=None):
    """auc_only: optional dict model -> list of subsets for which that model's AUC is drawn (default all)."""
    rng = np.random.default_rng(seed)
    firms, inv = np.unique(firmv, return_inverse=True)
    auc_only = auc_only or {}
    eng = {(e, s): AUCEngine(yv[m], p[m]) for e, p in preds.items() for s, m in subs.items()
           if s in auc_only.get(e, subs)}
    L = {e: row_loss(yv, p) for e, p in preds.items()}
    ll_subsets = ll_subsets if ll_subsets is not None else list(subs)
    out = {}
    for b in range(n_boot):
        cnt = np.bincount(rng.integers(0, len(firms), len(firms)), minlength=len(firms))
        w = cnt[inv].astype(float)
        for s, m in subs.items():
            ws = w[m]
            ok = (ws * yv[m]).sum() > 0 and (ws * (1 - yv[m])).sum() > 0
            for e in preds:
                if (e, s) in eng:
                    out.setdefault(("auc", e, s), []).append(eng[(e, s)](ws) if ok else np.nan)
                if s in ll_subsets:
                    out.setdefault(("log_loss", e, s), []).append(float((ws * L[e][m]).sum() / ws.sum()))
        if b % 100 == 0:
            log("bootstrap draw", b)
    return {k: np.asarray(v) for k, v in out.items()}


def ci(d):
    d = d[np.isfinite(d)]
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


# equivalence check vs the shared function (40 draws, 15,000-firm subsample), GAM-01 vs GB-01
rng0 = np.random.default_rng(123)
sub_f = rng0.choice(np.unique(firm), 15000, replace=False)
ms = np.isin(firm, sub_f)
ref = M.paired_bootstrap(y[ms], P["GAM-01"][ms], P["GB-01"][ms], firm[ms], n_boot=40, seed=0)
dr = bootstrap_draws(y[ms], firm[ms], {"GAM-01": P["GAM-01"][ms], "GB-01": P["GB-01"][ms]},
                     {"ALL": np.ones(ms.sum(), bool)}, n_boot=40, seed=0)
fast = {"d_auc": dr[("auc", "GAM-01", "ALL")] - dr[("auc", "GB-01", "ALL")],
        "d_log_loss": dr[("log_loss", "GAM-01", "ALL")] - dr[("log_loss", "GB-01", "ALL")]}
eq = pd.DataFrame([{"metric": r["metric"], "shared_lo": r["ci_low"], "fast_lo": np.percentile(fast[r["metric"]], 2.5),
                    "shared_hi": r["ci_high"], "fast_hi": np.percentile(fast[r["metric"]], 97.5)} for _, r in ref.iterrows()])
eq["max_abs_diff"] = np.maximum(abs(eq.shared_lo - eq.fast_lo), abs(eq.shared_hi - eq.fast_hi))
eq.to_csv(OUT / "r1b_bootstrap_equivalence.csv", index=False)
assert (eq["max_abs_diff"] < 1e-9).all()
log("equivalence ok")

BOOT_MODS = ["GB-01", "CAL-01", "SC-03", "RC-02", "GAM-01", "GAM-01+CAL", "GAM-02", "GAM-02+CAL"]
LL_SUBS = ["ALL|ALL", "year|2008", "year|2009", "year|2010", "firm|seen", "firm|new"]
draws = bootstrap_draws(y, firm, {e: P[e] for e in BOOT_MODS}, subsets, ll_subsets=LL_SUBS,
                        auc_only={e: LL_SUBS for e in ("GAM-01", "GAM-02", "GAM-02+CAL")})
np.savez_compressed(OUT / "r1b_bootstrap_draws.npz", **{"|".join(k).replace("/", "_"): v for k, v in draws.items()})
log("bootstrap done")

# 1d. paired comparisons for the GAMs
PAIRS = [("GAM-01", "GB-01"), ("GAM-01", "SC-03"), ("GAM-01", "RC-02"), ("GAM-02", "GB-01"), ("GAM-02", "SC-03"),
         ("GAM-02", "GAM-01"), ("GAM-01+CAL", "CAL-01"), ("GAM-02+CAL", "CAL-01"), ("GAM-02+CAL", "GAM-01+CAL")]
prow = []
for a, b in PAIRS:
    for s in LL_SUBS:
        m = subsets[s]
        r = {"new": a, "old": b, "subset": s, "n": int(m.sum()), "defaults": int(y[m].sum())}
        r["d_auc"] = M.auc(y[m], P[a][m]) - M.auc(y[m], P[b][m])
        d = draws[("auc", a, s)] - draws[("auc", b, s)]
        r["d_auc_lo"], r["d_auc_hi"] = ci(d)
        r["d_auc_share_fav_new"] = float(np.nanmean(d > 0))
        r["d_log_loss"] = M.log_loss(y[m], P[a][m]) - M.log_loss(y[m], P[b][m])
        d = draws[("log_loss", a, s)] - draws[("log_loss", b, s)]
        r["d_log_loss_lo"], r["d_log_loss_hi"] = ci(d)
        r["d_log_loss_share_fav_new"] = float(np.mean(d < 0))
        r["n_draws"] = len(d)
        prow.append(r)
pd.DataFrame(prow).to_csv(OUT / "r1b_gam_paired_bootstrap.csv", index=False)

# ------------------------------------------------------------------------------------------------
# 2. Power and level by segment
# ------------------------------------------------------------------------------------------------
SEG_MODS = ["GB-01", "GAM-01+CAL", "RC-02", "SC-03"]
segr = []
for s, m in subsets.items():
    g, c = s.split("|")
    n, k = int(m.sum()), int(y[m].sum())
    lo_o, hi_o = cp_ci(k, n)
    for e in SEG_MODS + ["CAL-01", "GAM-01"]:
        r = {"dimension": g, "cell": c, "model": e, "n": n, "defaults": k, "observed_dr": k / n,
             "dr_ci_low": lo_o, "dr_ci_high": hi_o, "mean_pd": float(P[e][m].mean())}
        r["obs_over_pred"] = r["observed_dr"] / r["mean_pd"]
        r["mean_pd_inside_dr_ci"] = bool(lo_o <= r["mean_pd"] <= hi_o)
        a = M.auc(y[m], P[e][m])
        r["auc"], r["ar"] = a, 2 * a - 1
        if ("auc", e, s) in draws:
            r["auc_lo"], r["auc_hi"] = ci(draws[("auc", e, s)])
            r["ar_lo"], r["ar_hi"] = 2 * r["auc_lo"] - 1, 2 * r["auc_hi"] - 1
        if e != "GB-01" and ("auc", e, s) in draws:
            d = draws[("auc", "GB-01", s)] - draws[("auc", e, s)]
            r["gb01_minus_model_auc"] = M.auc(y[m], P["GB-01"][m]) - a
            r["gb01_minus_model_lo"], r["gb01_minus_model_hi"] = ci(d)
        segr.append(r)
segr = pd.DataFrame(segr)
segr.to_csv(OUT / "r1b_segment_power.csv", index=False)
log("segments done")

# ------------------------------------------------------------------------------------------------
# 3. Missing-input check (L1): debt_split_missing and core inputs missing in the raw data
# ------------------------------------------------------------------------------------------------
CORE_NUM = [c for c in B.CORE_FEATURES if c in B.NUM_FEATURES]
CORE_RAW = {"log_assets": ["asst_tot"], "equity_ratio": ["eqty_tot", "asst_tot"], "st_debt_ratio": ["debt_st", "asst_tot"],
            "cash_ratio": ["cash_and_equiv", "asst_tot"], "roa_net": ["profit", "asst_tot"],
            "interest_coverage": ["ebitda", "exp_financing"], "interest_to_assets": ["exp_financing", "asst_tot"],
            "ar_days": ["AR", "rev_operating"], "tangible_ratio": ["asst_tang_fixed", "asst_tot"], "neg_equity": ["eqty_tot"]}
raw_fields = sorted({x for v in CORE_RAW.values() for x in v})
rawc = B.coerce_input(base)
any_raw_core_missing = rawc[raw_fields].isna().any(axis=1).to_numpy()
feat = []
for f in splits["folds"]:
    m = fold == f["fold"]
    fp = B.load_params(str(ROOT / f"experiments/gam/GAM-01/model_fold{f['fold']}/feature_params.json"))
    X = B.transform(base.loc[m], fp, impute=False)
    feat.append(pd.DataFrame({"debt_split_missing": X["debt_split_missing"].to_numpy(),
                              "n_core_nan": X[CORE_NUM].isna().sum(axis=1).to_numpy(),
                              "ateco_group": X["ateco_group"].astype(str).to_numpy()}, index=np.flatnonzero(m)))
feat = pd.concat(feat).sort_index()
dsm = feat["debt_split_missing"].to_numpy() == 1
core_nan = feat["n_core_nan"].to_numpy() > 0
MISS = {"debt_split_missing=1": dsm, "debt_split_missing=0": ~dsm,
        "core_feature_NaN_after_rebuild (median-filled in GLM view)": core_nan,
        "no_core_feature_NaN": ~core_nan,
        "core_raw_input_missing_in_raw_data": any_raw_core_missing, "no_core_raw_input_missing": ~any_raw_core_missing}
mr = []
for gname, gm in MISS.items():
    for t in ("ALL", 2008, 2009, 2010):
        m = gm & ((year == t) if t != "ALL" else True)
        ym = (year == t) if t != "ALL" else np.ones(len(y), bool)
        n, k = int(m.sum()), int(y[m].sum())
        if n == 0:
            continue
        lo, hi = cp_ci(k, n)
        for e in ("GB-01", "CAL-01", "GAM-01+CAL", "RC-02", "SC-03"):
            mp = float(P[e][m].mean())
            ratio_all = y[ym].mean() / P[e][ym].mean()
            mr.append({"group": gname, "test_year": t, "model": e, "n": n, "share_of_rows": n / ym.sum(), "defaults": k,
                       "observed_dr": k / n, "dr_ci_low": lo, "dr_ci_high": hi, "mean_pd": mp, "obs_over_pred": k / n / mp,
                       "obs_over_pred_all_rows_same_year": ratio_all,
                       "relative_obs_over_pred": (k / n / mp) / ratio_all,
                       "mean_pd_inside_dr_ci": bool(lo <= mp <= hi),
                       "auc_within_group": M.auc(y[m], P[e][m])})
pd.DataFrame(mr).to_csv(OUT / "r1b_missing_input_check.csv", index=False)
log("missing check done")

# ------------------------------------------------------------------------------------------------
# 4. Grade grid: edges from training-year predictions (fold-3 GB-01 on 2007-2009, CAL-01 level)
# ------------------------------------------------------------------------------------------------
gtr = pd.read_parquet(ROOT / "experiments/calibration/cache/gb_train_3.parquet")
assert gtr["fs_year"].max() <= 2009
m3 = fold == 3
dl = logit(P["CAL-01"][m3]) - logit(P["GB-01"][m3])
unc = (P["CAL-01"][m3] > 0.00031) & (P["CAL-01"][m3] < 0.98)
delta3 = float(np.median(dl[unc]))
ct = pd.read_csv(ROOT / "reports/calibration/ct_targets.csv")
tau10 = float(ct[(ct.label == "s6") & (ct.rule == "e_last_6m_annualised") & (ct.test_year == 2010)].target.iloc[0])
f3 = [x for x in splits["folds"] if x["fold"] == 3][0]
rw3 = float(panel[panel["fs_year"].isin(f3["train_years"]) & (panel["label_status"] == "ok")]["y"].mean())
delta_check = float(logit(tau10) - logit(rw3))
gb_tr_pd = np.clip(sigmoid(logit(gtr["pd"].to_numpy()) + delta3), 0.0003, 0.99)
# sanity: the training-row GB-01 pd (anchored) reproduces the OOT model? (only that it is in-sample fold 3)
OLD = [0.0003, 0.001, 0.0025, 0.005, 0.0075, 0.01, 0.015, 0.025, 0.05, 0.10, 0.20, 0.99]


def nice(x):
    """round a PD to 2 significant digits (in %), e.g. 0.001634 -> 0.0016"""
    d = int(np.floor(np.log10(x))) - 1
    return float(np.round(x, -d))


def assign(p, edges):
    return np.clip(np.searchsorted(np.asarray(edges[1:-1]), p, side="right") + 1, 1, len(edges) - 1)


edges = list(OLD)
CAP_SHARE = 0.12
for _ in range(50):
    g = assign(gb_tr_pd, edges)
    share = np.bincount(g, minlength=len(edges))[1:] / len(g)
    j = int(np.argmax(share))
    if share[j] <= CAP_SHARE:
        break
    lo_e, hi_e = edges[j], edges[j + 1]
    inside = gb_tr_pd[(g == j + 1)]
    cut = nice(np.median(inside))
    if not (lo_e < cut < hi_e):
        cut = nice(np.sqrt(lo_e * hi_e))
    edges = sorted(edges + [cut])
NEW = edges


def grade_table(p, yy, edges, tag):
    g = assign(p, edges)
    rows = []
    for j in range(1, len(edges)):
        m = g == j
        n, k = int(m.sum()), int(yy[m].sum()) if yy is not None else 0
        r = {"table": tag, "grade": j, "pd_band": f"[{edges[j-1]*100:.3g}%, {edges[j]*100:.3g}%)", "lower": edges[j - 1],
             "upper": edges[j], "n": n, "share_of_firms": n / len(p)}
        if n:
            r["mean_pd"] = float(p[m].mean())
        if yy is not None and n:
            lo, hi = cp_ci(k, n)
            exp_k = float(p[m].sum())
            # acceptance interval for the default COUNT given the PDs (Poisson-binomial, normal approx;
            # binomial(n, mean_pd) quantiles used when the expected count is < 20)
            if exp_k < 20:
                a_lo, a_hi = stats.binom.ppf([0.025, 0.975], n, r["mean_pd"])
            else:
                sd = np.sqrt((p[m] * (1 - p[m])).sum())
                a_lo, a_hi = exp_k - 1.96 * sd, exp_k + 1.96 * sd
            r.update({"defaults": k, "share_of_defaults": k / yy.sum(), "observed_dr": k / n, "dr_ci_low": lo, "dr_ci_high": hi,
                      "expected_defaults": exp_k, "accept_lo_defaults": float(a_lo), "accept_hi_defaults": float(a_hi),
                      "excess_defaults": k - exp_k, "obs_over_pred": (k / n) / r["mean_pd"],
                      "mean_pd_inside_dr_ci": bool(lo <= r["mean_pd"] <= hi),
                      "flag": "UNDER-predicted" if k > a_hi else ("OVER-predicted" if k < a_lo else "")})
        rows.append(r)
    return pd.DataFrame(rows)


y10 = y[year == 2010]
p10 = P["CAL-01"][year == 2010]
oot0809 = P["CAL-01"][(year == 2008) | (year == 2009)]
gt = pd.concat([
    grade_table(p10, y10, OLD, "old_11_grades_2010_CAL-01 (reproduces grade_table_2010.csv)"),
    grade_table(gb_tr_pd, None, NEW, "new_grid_training_years_2007-2009_fold3_insample (edge-setting sample)"),
    grade_table(oot0809, None, NEW, "new_grid_OOT_2008-2009_CAL-01 (share check)"),
    grade_table(p10, y10, NEW, "new_grid_2010_CAL-01"),
    grade_table(P["GB-01"][year == 2010], y10, NEW, "new_grid_2010_GB-01_own_level"),
])
gt.to_csv(OUT / "r1b_grade_table.csv", index=False)
json.dump({"edges_old": OLD, "edges_new": NEW, "cap_share": CAP_SHARE,
           "edge_sample": "GB-01 fold-3 model (fit 2007-2009) in-sample PDs on its 475,861 training rows, shifted by the fold-3 CAL-01 delta",
           "delta_fold3_from_files": delta3, "delta_fold3_logit_tau_minus_logit_rw": delta_check, "tau_rule_e_2010": tau10,
           "r_w_fold3": rw3, "rule": "split the grade with the largest share > 12% at the 2-significant-digit rounded median PD of its rows; repeat"},
          open(OUT / "r1b_grade_edges.json", "w"), indent=1)
log("grades done", NEW)

# ------------------------------------------------------------------------------------------------
# 5. Overfitting evidence: in-sample (training rows) vs out-of-time AUC, per fold
# ------------------------------------------------------------------------------------------------
ov = []
for f in splits["folds"]:
    k = f["fold"]
    tr = panel[panel["fs_year"].isin(f["train_years"]) & (panel["label_status"] == "ok")].reset_index(drop=True)
    m = fold == k
    g = pd.read_parquet(ROOT / f"experiments/calibration/cache/gb_train_{k}.parquet")
    rc = pd.read_parquet(ROOT / f"experiments/calibration/cache/rc_train_{k}.parquet")
    pg = GL.predict_pd(MODELS[("GAM-01", k)], tr.drop(columns=LABEL_COLS))
    for e, yy, pp in (("GB-01", g["y"].to_numpy(), g["pd"].to_numpy()), ("RC-02", rc["y"].to_numpy(), rc["pd"].to_numpy()),
                      ("GAM-01", tr["y"].to_numpy(), pg)):
        ins = M.auc(yy, pp)
        oo = M.auc(y[m], P[e][m])
        last = None
        if e == "GAM-01":
            lm = (tr["fs_year"] == max(f["train_years"])).to_numpy()
            last = M.auc(yy[lm], pp[lm])
        elif e == "GB-01":
            lm = (g["fs_year"] == max(f["train_years"])).to_numpy()
            last = M.auc(yy[lm], pp[lm])
        ov.append({"model": e, "fold": k, "train_years": str(f["train_years"]), "test_year": f["test_year"],
                   "n_train_scored": len(yy), "auc_in_sample_all_train": ins, "auc_in_sample_last_train_year": last,
                   "auc_oot": oo, "gap_in_minus_oot": ins - oo, "ar_gap_points": 2 * (ins - oo) * 100,
                   "cal_slope_oot": M.calibration_slope_intercept(y[m], P[e][m])["cal_slope"]})
pd.DataFrame(ov).to_csv(OUT / "r1b_insample_vs_oot.csv", index=False)

# selection-risk ledger: how many prediction vectors have been scored on each test year
mb = pd.read_csv(ROOT / "reports/calibration/methods_by_year.csv")
n_files = len(list((ROOT / "experiments").glob("*/*/oot_predictions*.parquet")))
n_cal_cfg = mb[["model", "rule", "method"]].drop_duplicates().shape[0]
json.dump({"oot_prediction_files_covering_2008_2010": n_files, "calibration_configs_scored_on_2008_2010": n_cal_cfg},
          open(OUT / "r1b_selection_ledger.json", "w"), indent=1)

# ------------------------------------------------------------------------------------------------
# Figures
# ------------------------------------------------------------------------------------------------
fig, axs = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)
for ax, dim in zip(axs, ("size", "ateco", "year")):
    sd = segr[(segr.dimension == dim) & segr.model.isin(SEG_MODS)]
    cells = list(dict.fromkeys(sd["cell"]))
    for i, e in enumerate(SEG_MODS):
        d = sd[sd.model == e].set_index("cell").loc[cells]
        x = np.arange(len(cells)) + (i - 1.5) * 0.18
        ax.errorbar(x, d["auc"], yerr=[np.clip(d["auc"] - d["auc_lo"], 0, None), np.clip(d["auc_hi"] - d["auc"], 0, None)], fmt="o", ms=3, lw=0.8, label=e)
    ax.set_xticks(np.arange(len(cells)))
    ax.set_xticklabels([f"{c}\n({int(sd[sd.cell == c].defaults.iloc[0])}d)" for c in cells], fontsize=6, rotation=90 if dim == "ateco" else 0)
    ax.set_title(f"AUC by {dim} (firm-clustered 95% CI; defaults in brackets)")
axs[0].legend(fontsize=7)
axs[0].set_ylabel("AUC (AR = 2 AUC - 1)")
plt.tight_layout()
plt.savefig(FIG / "r1b_segment_auc.png", dpi=130)
plt.close()

g10 = gt[gt.table == "new_grid_2010_CAL-01"]
fig, ax = plt.subplots(figsize=(7, 5))
ax.errorbar(g10["mean_pd"], g10["observed_dr"], yerr=[g10["observed_dr"] - g10["dr_ci_low"], g10["dr_ci_high"] - g10["observed_dr"]],
            fmt="o", ms=4, label="2010 observed (Clopper-Pearson 95%)")
ax.plot([2e-4, 0.6], [2e-4, 0.6], "k:", lw=0.8)
for _, r in g10.iterrows():
    ax.annotate(str(r["grade"]), (r["mean_pd"], r["observed_dr"]), fontsize=7, xytext=(3, -8), textcoords="offset points")
ax.set(xscale="log", yscale="log", xlabel="mean PD in grade (CAL-01)", ylabel="observed default rate 2010",
       title="Re-split grade grid (edges set on 2007-2009), 2010 outcomes")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(FIG / "r1b_grade_2010.png", dpi=130)
plt.close()
log("all done")
