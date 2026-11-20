"""Render reports/tournament/r1b_diagnostics.md from r1b_diagnostics_template.md + the r1b_*.csv files.
Placeholders {{T:name}} are replaced by markdown tables built below; every number comes from a CSV."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent


def md(df, fmt=None):
    fmt = fmt or {}
    d = df.copy()
    for c in d.columns:
        if c in fmt:
            d[c] = d[c].map(lambda v, f=fmt[c]: "" if pd.isna(v) else f.format(v))
    return d.to_markdown(index=False)


T = {}
met = pd.read_csv(OUT / "r1b_metrics_long.csv")
T["contract"] = md(pd.read_csv(OUT / "r1b_contract.csv")[["exp", "file", "n_rows", "missing_test_rows", "dup_row_fold", "nan_pd",
                                                          "pd_outside_open_01", "min_pd", "max_pd", "PASS"]],
                   {"min_pd": "{:.5f}", "max_pd": "{:.4f}"})
rp = pd.read_csv(OUT / "r1b_gam_repro.csv")
T["repro"] = md(rp[["exp", "fold", "n_test", "max_abs_diff_own", "max_abs_diff_raw", "max_abs_diff_cal", "r_w_config", "r_w_recomputed",
                    "n_train_config", "n_train_recomputed", "tau_rule_e", "delta", "chosen_cfg", "predict_secs_2threads"]],
                {"r_w_config": "{:.6f}", "r_w_recomputed": "{:.6f}", "tau_rule_e": "{:.5f}", "delta": "{:.4f}"})

lead = met[met.group.isin(["ALL", "2008", "2009", "2010", "seen", "new"])]
draws = np.load(OUT / "r1b_bootstrap_draws.npz")


def auc_ci(e, s):
    k = f"auc|{e}|{s}"
    if k not in draws:
        return ""
    d = draws[k]
    d = d[np.isfinite(d)]
    return f"[{np.percentile(d, 2.5):.4f}, {np.percentile(d, 97.5):.4f}]"


SUBMAP = {"ALL": "ALL|ALL", "2008": "year|2008", "2009": "year|2009", "2010": "year|2010", "seen": "firm|seen", "new": "firm|new"}
L = lead.copy()
L["auc_95ci"] = [auc_ci(e, SUBMAP[g]) for e, g in zip(L.exp, L.group)]
T["leader"] = md(L[L.group == "ALL"][["exp", "n", "defaults", "auc", "auc_95ci", "ar", "log_loss", "brier", "mean_pd", "observed_dr",
                                       "ratio_obs_to_pred", "cal_intercept", "cal_slope"]],
                 {"auc": "{:.4f}", "ar": "{:.3f}", "log_loss": "{:.5f}", "brier": "{:.5f}", "mean_pd": "{:.4%}", "observed_dr": "{:.4%}",
                  "ratio_obs_to_pred": "{:.3f}", "cal_intercept": "{:.3f}", "cal_slope": "{:.3f}"})
T["byyear"] = md(L[L.group != "ALL"][["exp", "group", "n", "defaults", "auc", "auc_95ci", "log_loss", "ratio_obs_to_pred", "cal_slope"]],
                 {"auc": "{:.4f}", "log_loss": "{:.5f}", "ratio_obs_to_pred": "{:.3f}", "cal_slope": "{:.3f}"})
pb = pd.read_csv(OUT / "r1b_gam_paired_bootstrap.csv")
pb["dAUC [95% CI]"] = pb.apply(lambda r: f"{r.d_auc:+.4f} [{r.d_auc_lo:+.4f}, {r.d_auc_hi:+.4f}]", axis=1)
pb["dLogLoss [95% CI]"] = pb.apply(lambda r: f"{r.d_log_loss:+.5f} [{r.d_log_loss_lo:+.5f}, {r.d_log_loss_hi:+.5f}]", axis=1)
T["pairs"] = md(pb[["new", "old", "subset", "defaults", "dAUC [95% CI]", "dLogLoss [95% CI]", "n_draws"]])
fv = pd.read_csv(OUT / "r1b_gam_face_validity.csv")
mfv = fv[fv.n_bins > 0]
piv = mfv[mfv.expected_sign != 0].pivot_table(index="term", columns=["exp", "fold"], values="against_share", aggfunc="first")
piv.columns = [f"{a} f{b}" for a, b in piv.columns]
piv = piv.reset_index()
sg = mfv.drop_duplicates("term").set_index("term")["expected_sign"]
piv.insert(1, "expected", piv["term"].map(sg))
T["fv"] = md(piv, {c: "{:.3f}" for c in piv.columns if c.startswith("GAM")})
f3 = mfv[(mfv.exp == "GAM-01") & (mfv.fold == 3)][["term", "expected_sign", "range_logodds", "lowest_bin_score", "highest_bin_score",
                                                   "missing_bin_score", "against_share"]]
T["fv3"] = md(f3, {"range_logodds": "{:.3f}", "lowest_bin_score": "{:+.3f}", "highest_bin_score": "{:+.3f}", "missing_bin_score": "{:+.3f}",
                   "against_share": "{:.3f}"})

seg = pd.read_csv(OUT / "r1b_segment_power.csv")


def segtab(dim):
    s = seg[(seg.dimension == dim) & seg.model.isin(["GB-01", "GAM-01+CAL", "RC-02", "SC-03"])].copy()
    s["AUC [CI]"] = s.apply(lambda r: f"{r.auc:.3f} [{r.auc_lo:.3f}, {r.auc_hi:.3f}]", axis=1)
    s["AR"] = s["ar"].map("{:.3f}".format)
    w = s.pivot_table(index=["cell", "n", "defaults"], columns="model", values="AUC [CI]", aggfunc="first").reset_index()
    a = s.pivot_table(index="cell", columns="model", values="AR", aggfunc="first").add_prefix("AR ").reset_index()
    w = w.merge(a, on="cell")
    gb = seg[(seg.dimension == dim) & (seg.model == "GAM-01+CAL")][["cell", "gb01_minus_model_auc", "gb01_minus_model_lo", "gb01_minus_model_hi"]]
    gb["GB-01 minus GAM-01+CAL"] = gb.apply(lambda r: f"{r.gb01_minus_model_auc:+.3f} [{r.gb01_minus_model_lo:+.3f}, {r.gb01_minus_model_hi:+.3f}]", axis=1)
    w = w.merge(gb[["cell", "GB-01 minus GAM-01+CAL"]], on="cell")
    return md(w)


for dim in ("year", "size", "ateco", "firm"):
    T[f"seg_{dim}"] = segtab(dim)
cal = seg[seg.model.isin(["GB-01", "CAL-01", "GAM-01+CAL"]) & (seg.dimension != "ALL")].copy()
cal["obs DR [CI]"] = cal.apply(lambda r: f"{r.observed_dr:.3%} [{r.dr_ci_low:.3%}, {r.dr_ci_high:.3%}]", axis=1)
cw = cal.pivot_table(index=["dimension", "cell", "n", "defaults", "obs DR [CI]"], columns="model", values="mean_pd", aggfunc="first").reset_index()
cr = cal.pivot_table(index=["dimension", "cell"], columns="model", values="obs_over_pred", aggfunc="first").add_prefix("obs/pred ").reset_index()
cw = cw.merge(cr, on=["dimension", "cell"])
T["seg_cal"] = md(cw, {"GB-01": "{:.3%}", "CAL-01": "{:.3%}", "GAM-01+CAL": "{:.3%}", "obs/pred GB-01": "{:.2f}",
                       "obs/pred CAL-01": "{:.2f}", "obs/pred GAM-01+CAL": "{:.2f}"})
T["size_edges"] = md(pd.read_csv(OUT / "r1b_size_quintile_edges.csv"), {c: "{:,.0f}" for c in ["q1_edge_eur", "q2_edge_eur", "q3_edge_eur", "q4_edge_eur"]})

mi = pd.read_csv(OUT / "r1b_missing_input_check.csv")
mi = mi[mi.model.isin(["GB-01", "CAL-01"])]
T["missing"] = md(mi[["group", "test_year", "model", "n", "defaults", "observed_dr", "dr_ci_low", "dr_ci_high", "mean_pd", "obs_over_pred",
                      "obs_over_pred_all_rows_same_year", "relative_obs_over_pred"]],
                  {"observed_dr": "{:.3%}", "dr_ci_low": "{:.3%}", "dr_ci_high": "{:.3%}", "mean_pd": "{:.3%}", "obs_over_pred": "{:.2f}",
                   "obs_over_pred_all_rows_same_year": "{:.2f}", "relative_obs_over_pred": "{:.2f}"})
mt = pd.read_csv(OUT / "r1b_missing_tests.csv")
T["missing_tests"] = md(mt, {"expected_at_model_level": "{:.1f}", "obs_over_pred": "{:.2f}", "expected_after_year_level_adjustment": "{:.1f}",
                             "relative_obs_over_pred": "{:.2f}", "relative_ci_low": "{:.2f}", "relative_ci_high": "{:.2f}",
                             "poisson_p_two_sided": "{:.3f}"})

g = pd.read_csv(OUT / "r1b_grade_table.csv")
gfmt = {"share_of_firms": "{:.1%}", "share_of_defaults": "{:.1%}", "mean_pd": "{:.3%}", "observed_dr": "{:.3%}", "dr_ci_low": "{:.3%}",
        "dr_ci_high": "{:.3%}", "expected_defaults": "{:.1f}", "accept_lo_defaults": "{:.0f}", "accept_hi_defaults": "{:.0f}",
        "excess_defaults": "{:+.1f}", "obs_over_pred": "{:.2f}", "defaults": "{:.0f}"}
cols = ["grade", "pd_band", "n", "share_of_firms", "defaults", "share_of_defaults", "mean_pd", "observed_dr", "dr_ci_low", "dr_ci_high",
        "expected_defaults", "accept_lo_defaults", "accept_hi_defaults", "excess_defaults", "obs_over_pred", "flag"]
for tag, key in (("new_grid_2010_CAL-01", "grade_new_2010"), ("new_grid_2010_GB-01_own_level", "grade_new_2010_gb"),
                 ("old_11_grades_2010_CAL-01 (reproduces grade_table_2010.csv)", "grade_old_2010")):
    T[key] = md(g[g.table == tag][cols].fillna({"flag": ""}), gfmt)
sh = g[g.table.str.startswith("new_grid")].pivot_table(index=["grade", "pd_band"], columns="table", values="share_of_firms", aggfunc="first").reset_index()
sh.columns = [c.split(" ")[0] if isinstance(c, str) else c for c in sh.columns]
T["grade_shares"] = md(sh.sort_values("grade"), {c: "{:.1%}" for c in sh.columns if c.startswith("new_grid")})
T["grade_summary"] = md(pd.read_csv(OUT / "r1b_grade_summary.csv"), {"max_share": "{:.1%}", "observed_defaults": "{:.0f}", "expected_defaults": "{:.1f}",
                                                                      "obs_over_pred_total": "{:.3f}", "excess_defaults_in_under_grades": "{:.1f}",
                                                                      "share_firms_in_under_grades": "{:.1%}"})
ge = json.load(open(OUT / "r1b_grade_edges.json"))
T["grade_edges"] = "old: " + ", ".join(f"{x*100:g}%" for x in ge["edges_old"]) + "\n\nnew: " + ", ".join(f"{x*100:g}%" for x in ge["edges_new"])

T["insample"] = md(pd.read_csv(OUT / "r1b_insample_vs_oot.csv"), {"auc_in_sample_all_train": "{:.4f}", "auc_in_sample_last_train_year": "{:.4f}",
                                                                  "auc_oot": "{:.4f}", "gap_in_minus_oot": "{:+.4f}", "ar_gap_points": "{:+.1f}",
                                                                  "cal_slope_oot": "{:.3f}"})
T["sameyear"] = md(pd.read_csv(OUT / "r1b_same_year_overfit.csv"), {"auc_in_sample": "{:.4f}", "auc_oot": "{:.4f}", "gap_auc": "{:+.4f}",
                                                                    "gap_ar_points": "{:+.1f}"})
T["stability"] = md(pd.read_csv(OUT / "r1b_stability_summary.csv"), {c: "{:.4f}" for c in ["auc_pooled", "auc_ci_width_pooled", "auc_2008", "auc_2009",
                                                                                           "auc_2010", "auc_range_years", "seen_minus_new_auc"]}
                    | {c: "{:.3f}" for c in ["cal_slope_pooled", "cal_slope_min_year", "cal_slope_max_year", "cal_slope_new_firms",
                                             "obs_over_pred_2008", "obs_over_pred_2010"]})
T["ledger"] = md(pd.DataFrame([json.load(open(OUT / "r1b_selection_ledger.json"))]))
T["equiv"] = md(pd.read_csv(OUT / "r1b_bootstrap_equivalence.csv"))

tpl = (OUT / "r1b_diagnostics_filled_template.md").read_text()
missing = set(re.findall(r"\{\{T:(\w+)\}\}", tpl)) - set(T)
assert not missing, missing
(OUT / "r1b_diagnostics.md").write_text(re.sub(r"\{\{T:(\w+)\}\}", lambda m_: T[m_.group(1)], tpl))
print("written", len(T), "tables")
