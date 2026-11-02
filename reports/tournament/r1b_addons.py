"""Referee round 1b add-ons (run after r1b_diagnostics.py). LOCKBOX: panel read with fs_year < 2011.

Writes r1b_same_year_overfit.csv, r1b_grade_summary.csv, r1b_missing_tests.csv, r1b_stability_summary.csv.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.common import metrics as M  # noqa: E402

OUT = ROOT / "reports/tournament"
splits = json.load(open(ROOT / "data/splits/walkforward.json"))
testrows = pd.read_parquet(ROOT / "data/splits/walkforward_test_rows.parquet")
panel = pd.read_parquet(ROOT / "data/interim/panel.parquet", columns=["row_id", "id", "fs_year", "y", "label_status"],
                        filters=[("fs_year", "<", 2011)])
base = testrows.merge(panel, on="row_id", how="left", validate="one_to_one")
y = base["y"].astype(int).to_numpy()
year = base["test_year"].to_numpy()

# 1. same-year in-sample vs out-of-time AUC: year t scored by the fold whose training includes t
#    (in-sample) vs by the fold for which t is the test year (OOT). Same rows? In-sample uses all ok rows
#    of year t; OOT uses the test rows of year t (= all ok rows of year t). Identical row sets.
oot = {e: pd.read_parquet(ROOT / f).merge(base[["row_id", "fold", "y", "test_year"]], on=["row_id", "fold"])
       for e, f in (("GB-01", "experiments/gbm/GB-01/oot_predictions.parquet"),
                    ("RC-02", "experiments/riskcalc/RC-02/oot_predictions.parquet"))}
rows = []
for t, k_in in ((2008, 2), (2008, 3), (2009, 3)):
    for e, cache in (("GB-01", "gb_train"), ("RC-02", "rc_train")):
        c = pd.read_parquet(ROOT / f"experiments/calibration/cache/{cache}_{k_in}.parquet")
        c = c[c["fs_year"] == t]
        o = oot[e][oot[e]["test_year"] == t]
        rows.append({"model": e, "year": t, "in_sample_fold": k_in, "n_in": len(c), "n_oot": len(o),
                     "auc_in_sample": M.auc(c["y"], c["pd"]), "auc_oot": M.auc(o["y"], o["pd"])})
# GAM-01 in-sample values come from the main script (refit-free: saved fold models on training rows)
ins = pd.read_csv(OUT / "r1b_insample_vs_oot.csv")
r = pd.DataFrame(rows)
r["gap_auc"] = r["auc_in_sample"] - r["auc_oot"]
r["gap_ar_points"] = 200 * r["gap_auc"]
r.to_csv(OUT / "r1b_same_year_overfit.csv", index=False)
print(r.round(4).to_string())

# 2. grade summary for 2010 (new grid, CAL-01 and GB-01 own level)
g = pd.read_csv(OUT / "r1b_grade_table.csv")
summ = []
for tag in g["table"].unique():
    d = g[g["table"] == tag]
    s = {"table": tag, "n_grades": len(d), "max_share": d["share_of_firms"].max()}
    if d["defaults"].notna().any():
        s.update({"observed_defaults": d["defaults"].sum(), "expected_defaults": d["expected_defaults"].sum(),
                  "obs_over_pred_total": d["defaults"].sum() / d["expected_defaults"].sum(),
                  "grades_under": ",".join(str(int(x)) for x in d.loc[d["flag"] == "UNDER-predicted", "grade"]),
                  "grades_over": ",".join(str(int(x)) for x in d.loc[d["flag"] == "OVER-predicted", "grade"]),
                  "excess_defaults_in_under_grades": d.loc[d["flag"] == "UNDER-predicted", "excess_defaults"].sum(),
                  "share_firms_in_under_grades": d.loc[d["flag"] == "UNDER-predicted", "share_of_firms"].sum(),
                  "observed_dr_monotone": bool(np.all(np.diff(d["observed_dr"].to_numpy()) > 0))})
    summ.append(s)
pd.DataFrame(summ).to_csv(OUT / "r1b_grade_summary.csv", index=False)
print(pd.DataFrame(summ).round(4).to_string())

# 3. missing-input groups: is the group under-predicted MORE than the same year's rows overall?
#    expected defaults = sum over years of (n_group,t * mean_pd_group,t * obs/pred of all rows in t);
#    two-sided Poisson test of observed vs that expectation (diagnostic; ignores default correlation)
mi = pd.read_csv(OUT / "r1b_missing_input_check.csv")
tests = []
for (gname, model), d in mi[mi["test_year"] != "ALL"].groupby(["group", "model"]):
    exp_raw = float((d["n"] * d["mean_pd"]).sum())
    exp_adj = float((d["n"] * d["mean_pd"] * d["obs_over_pred_all_rows_same_year"]).sum())
    k = int(d["defaults"].sum())
    p = float(stats.poisson.sf(k - 1, exp_adj) * 2) if k > exp_adj else float(stats.poisson.cdf(k, exp_adj) * 2)
    lo, hi = stats.chi2.ppf(0.025, 2 * k) / 2 / exp_adj if k else 0.0, stats.chi2.ppf(0.975, 2 * k + 2) / 2 / exp_adj
    tests.append({"group": gname, "model": model, "n": int(d["n"].sum()), "defaults": k, "expected_at_model_level": exp_raw,
                  "obs_over_pred": k / exp_raw, "expected_after_year_level_adjustment": exp_adj,
                  "relative_obs_over_pred": k / exp_adj, "relative_ci_low": lo, "relative_ci_high": hi,
                  "poisson_p_two_sided": min(p, 1.0)})
pd.DataFrame(tests).to_csv(OUT / "r1b_missing_tests.csv", index=False)
print(pd.DataFrame(tests).round(4).to_string())

# 4. stability summary (point metrics + bootstrap CI widths)
met = pd.read_csv(OUT / "r1b_metrics_long.csv")
draws = np.load(OUT / "r1b_bootstrap_draws.npz")
st = []
for e in ["GB-01", "CAL-01", "SC-03", "RC-02", "GAM-01", "GAM-01+CAL", "GAM-02"]:
    d = met[met["exp"] == e].set_index("group")
    yrs = d.loc[["2008", "2009", "2010"]]
    key = f"auc|{e}|ALL|ALL"
    w = np.percentile(draws[key], 97.5) - np.percentile(draws[key], 2.5) if key in draws else np.nan
    st.append({"model": e, "auc_pooled": d.loc["ALL", "auc"], "auc_ci_width_pooled": w,
               "auc_2008": yrs.loc["2008", "auc"], "auc_2009": yrs.loc["2009", "auc"], "auc_2010": yrs.loc["2010", "auc"],
               "auc_range_years": yrs["auc"].max() - yrs["auc"].min(),
               "cal_slope_pooled": d.loc["ALL", "cal_slope"], "cal_slope_min_year": yrs["cal_slope"].min(),
               "cal_slope_max_year": yrs["cal_slope"].max(), "cal_slope_new_firms": d.loc["new", "cal_slope"],
               "obs_over_pred_2008": yrs.loc["2008", "ratio_obs_to_pred"], "obs_over_pred_2010": yrs.loc["2010", "ratio_obs_to_pred"],
               "seen_minus_new_auc": d.loc["seen", "auc"] - d.loc["new", "auc"]})
pd.DataFrame(st).to_csv(OUT / "r1b_stability_summary.csv", index=False)
print(pd.DataFrame(st).round(4).to_string())
