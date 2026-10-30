"""Stage 5 (calibration-specialist): 2011 pseudo-test, final-data targets, holdout level range,
label-window sensitivity, floor/cap, subgroup calibration, rating-grade table, figures and the
contract prediction files for the re-calibrated challengers.

Inputs: reports/calibration/{ct_targets,monthly_hazard,rates_by_cohort,scenario_optimum}.*,
experiments/calibration/cache/{final_scored_2010_2012.parquet, s3_preds.npz}.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.common import metrics as M  # noqa: E402

OUT = ROOT / "reports/calibration"
CACHE = ROOT / "experiments/calibration/cache"
EXP = ROOT / "experiments/calibration"
EPS = M.EPS
FLOOR, CAP = 0.0003, 0.99   # D-075 (a 50% cap was tested and rejected: floor_cap_effect.csv)


def logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def expit(z):
    return 1 / (1 + np.exp(-np.asarray(z, float)))


def kl(pi, q):
    return pi * np.log(pi / q) + (1 - pi) * np.log((1 - pi) / (1 - q))


def cp(k, n):
    ci = stats.binomtest(int(k), int(n)).proportion_ci(confidence_level=0.95, method="exact")
    return ci.low, ci.high


rates = pd.read_csv(OUT / "rates_by_cohort.csv")
haz = pd.read_csv(OUT / "monthly_hazard.csv")
part = pd.read_csv(OUT / "cohort2011_partial.csv")
opt = json.load(open(OUT / "scenario_optimum.json"))
R = {lab: rates[rates.label == lab].set_index("fs_year").rate for lab in ("s0", "s3", "s6")}
r11_est = float(part.loc[part.item.str.startswith("12-month rate, scaled by mean"), "value"].iloc[0])
r11_lo, r11_hi = [float(x) for x in part.loc[part.item.str.contains("range using"), "value"].iloc[0].split(" .. ")]
r11_11m = float(part.loc[part.item == "11-month rate", "value"].iloc[0])

# ------------------------------------------------------------------ 1. rules at the END of the data
def trend(yr: pd.Series):
    b1, b0 = np.polyfit(np.array(yr.index, float), logit(yr.to_numpy()), 1)
    return float(expit(b0 + b1 * (max(yr.index) + 1)))


s6 = R["s6"].copy()
s6_with11 = pd.concat([s6, pd.Series({2011: r11_est})])
ok = pd.read_parquet(ROOT / "data/interim/panel.parquet", columns=["fs_year", "y", "label_status"]).query("label_status=='ok'")
pooled_0710 = float(ok.y.mean())
# e at the end of the data: months 7-12 of the 2011 cohort's window; month 12 unobserved -> scale the
# observed months 7-11 by the average (7-12)/(7-11) hazard ratio of complete cohorts 2007-2010.
hz = haz.copy()
ratio = []
for c in range(2007, 2011):
    h = hz[hz.cohort == c].set_index("month_in_window").hazard
    ratio.append(h.loc[7:12].sum() / h.loc[7:11].sum())
h11 = hz[hz.cohort == 2011].set_index("month_in_window").hazard
e_end = float(2 * h11.loc[7:11].sum() * np.mean(ratio))
h10 = hz[hz.cohort == 2010].set_index("month_in_window").hazard
e_2010cohort = float(2 * h10.loc[7:12].sum())
final_rules = pd.DataFrame([
    {"rule": "a_pooled (2007-2010 complete labels)", "target": pooled_0710},
    {"rule": "b_last_complete_year (fs2010)", "target": float(s6.loc[2010])},
    {"rule": "b_last_year incl. fs2011 estimate (11 of 12 months, seasonally scaled)", "target": r11_est},
    {"rule": "c_trend_logit (2007-2010)", "target": trend(s6)},
    {"rule": "c_trend_logit (2007-2011 est.)", "target": trend(s6_with11)},
    {"rule": "c2_half_trend (2007-2010)", "target": float(expit(logit(s6.loc[2010]) + 0.5 * (logit(trend(s6)) - logit(s6.loc[2010]))))},
    {"rule": "d_ttc_equal_weight (2007-2011 est.)", "target": float(s6_with11.mean())},
    {"rule": "e_last_6m_annualised, latest complete half-year (Jan-Jun 2012, cohort 2010)", "target": e_2010cohort},
    {"rule": "e_last_6m_annualised, latest half-year (Dec 2012-May 2013, cohort 2011, seasonally scaled)", "target": e_end},
    {"rule": "f_mean_last2 (fs2010, fs2011 est.)", "target": float(s6_with11.iloc[-2:].mean())},
    {"rule": "scenario: expected-log-loss optimum over fs2008-2011 rates", "target": opt["expected_loss_optimum_equal_weights"]},
    {"rule": "scenario: minimax-regret optimum over fs2008-2011 rates", "target": opt["minimax_regret_optimum"]},
])
# ---- recommended level: PIT anchor (latest cohort) + log-loss convexity adjustment for an unknown horizon.
# Year-on-year changes of the cohort default rate on the logit scale (fs2007..fs2011 est.) give the
# volatility sigma of a driftless logit random walk (the persistence model behind rule b). If the holdout
# lies h years after the anchor, its rate is anchor * lognormal-ish noise; expected log loss is minimised by
# the EXPECTED rate, which exceeds the anchor (convexity). h is unknown: report h = 1, 2, 3; ship h = 2.
dz = np.diff(logit(s6_with11.to_numpy()))
sigma = float(np.sqrt(np.mean(dz ** 2)))
rng = np.random.default_rng(0)
eps = rng.standard_normal(400_000)
rw = []
for anchor_name, anchor in (("fs2011 cohort estimate", r11_est), ("e at end of data", e_end)):
    for h in (1, 2, 3):
        rw.append({"anchor": anchor_name, "anchor_rate": anchor, "horizon_years": h, "sigma_logit_per_year": sigma,
                   "expected_rate": float(expit(logit(anchor) + sigma * np.sqrt(h) * eps).mean()),
                   "p10_rate": float(expit(logit(anchor) + sigma * np.sqrt(h) * np.quantile(eps, 0.10))),
                   "p90_rate": float(expit(logit(anchor) + sigma * np.sqrt(h) * np.quantile(eps, 0.90)))})
rw = pd.DataFrame(rw)
rw.to_csv(OUT / "recommended_level_random_walk.csv", index=False)
TAU = float(rw.query("anchor == 'fs2011 cohort estimate' and horizon_years == 2").expected_rate.iloc[0])
final_rules = pd.concat([final_rules, pd.DataFrame([
    {"rule": "RECOMMENDED: fs2011 anchor + convexity, logit random walk, h = 2 years", "target": TAU}])])
final_rules.to_csv(OUT / "final_level_candidates.csv", index=False)

# ------------------------------------------------------------------ 2. 2011 pseudo-test (fold-4 model)
F = pd.read_parquet(CACHE / "final_scored_2010_2012.parquet")
F["alive"] = F.label_status.isin(["ok", "censored"])
r_w4 = pooled_0710
c11 = F[(F.fs_year == 2011) & F.alive].copy()
c11["y11"] = ((c11.def_date >= c11.stmt_date + pd.DateOffset(months=6)) &
              (c11.def_date <= pd.Timestamp("2013-05-31"))).astype(int)
# rules for test year 2011 from 2007-2010 labels only
t2011 = {"a_pooled": pooled_0710, "b_last_year": float(s6.loc[2010]), "c_trend_logit": trend(s6),
         "c2_half_trend": float(expit(logit(s6.loc[2010]) + 0.5 * (logit(trend(s6)) - logit(s6.loc[2010])))),
         "d_ttc_equal_weight": float(s6.mean()), "e_last_6m_annualised": e_2010cohort,
         "f_mean_last2": float(s6.iloc[-2:].mean()), "none (training-window anchor)": pooled_0710}
p11rows = []
for model, col in (("GB-01", "pd_gb"), ("RC-02", "pd_rc")):
    p = c11[col].to_numpy()
    for rule, tau in t2011.items():
        q = expit(logit(p) + logit(tau) - logit(r_w4))
        p11rows.append({"model": model, "rule": rule, "tau": tau, "mean_pd_2011_pop": q.mean(),
                        "est_12m_rate": r11_est, "est_range_low": r11_lo, "est_range_high": r11_hi,
                        "obs_over_pred": r11_est / q.mean(), "level_kl_1e4": kl(r11_est, q.mean()) * 1e4,
                        "auc_11m_label": M.auc(c11.y11, p)})
p11 = pd.DataFrame(p11rows)
# AUC / level on complete 2011 labels under s0 and s3 (fold-4 model never saw fs2011)
lw = []
for lab, yc, sc in (("s0", "y_s0", "status_s0"), ("s3", "y_s3", "status_s3")):
    g = F[(F.fs_year == 2011) & (F[sc] == "ok")]
    for model, col in (("GB-01", "pd_gb"), ("RC-02", "pd_rc")):
        lw.append({"label": lab, "model": model, "n": len(g), "defaults": int(g[yc].sum()),
                   "observed_rate": g[yc].mean(), "mean_pd_s6_anchor": g[col].mean(),
                   "auc": M.auc(g[yc], g[col])})
p11.to_csv(OUT / "q6_2011_pseudo_test.csv", index=False)
pd.DataFrame(lw).to_csv(OUT / "q5_2011_other_labels_fold4.csv", index=False)

# ------------------------------------------------------------------ 3. holdout level range
hl = []
for model, col in (("GB-01", "pd_gb"), ("RC-02", "pd_rc")):
    for t in (2010, 2011, 2012):
        p = F.loc[(F.fs_year == t) & F.alive, col].to_numpy()
        q = np.clip(expit(logit(p) + logit(TAU) - logit(r_w4)), FLOOR, CAP)
        hl.append({"model": model, "population_fs_year": t, "n": len(p), "mean_pd_anchor_2007_2010": p.mean(),
                   "mean_pd_recommended": q.mean(), "median_pd_recommended": float(np.median(q)),
                   "share_at_floor": float((q <= FLOOR).mean()), "share_at_cap": float((q >= CAP).mean())})
pd.DataFrame(hl).to_csv(OUT / "holdout_level_range.csv", index=False)

# ------------------------------------------------------------------ 4. label-window sensitivity
lab_rows = []
for lab in ("s0", "s3", "s6"):
    r = R[lab]
    r_post = (r.loc[2008:2011] if 2011 in r.index else pd.concat([r.loc[2008:2010], pd.Series({2011: r11_est})]))
    # same recommended rule as for s6: anchor = fs2011 rate, logit random walk, h = 2 years
    ser = pd.concat([r.loc[2007:2010], pd.Series({2011: float(r.loc[2011]) if 2011 in r.index else r11_est})])
    sig_l = float(np.sqrt(np.mean(np.diff(logit(ser.to_numpy())) ** 2)))
    tau_l = float(expit(logit(ser.loc[2011]) + sig_l * np.sqrt(2) * eps).mean())
    lab_rows.append({"label": lab, "rate_fs2010": float(r.loc[2010]),
                     "rate_fs2011": float(r.loc[2011]) if 2011 in r.index else r11_est,
                     "fs2011_is_estimate": 2011 not in r.index,
                     "pooled_2007_2010": float(ok.y.mean()) if lab == "s6" else np.nan,
                     "sigma_logit": sig_l, "recommended_tau_same_rule": tau_l, "ratio_to_s6_tau": np.nan})
lab_df = pd.DataFrame(lab_rows)
lab_df["ratio_to_s6_tau"] = lab_df.recommended_tau_same_rule / float(lab_df.query("label=='s6'").recommended_tau_same_rule.iloc[0])
cost = []
for _, a in lab_df.iterrows():          # a = level we ship
    for _, b in lab_df.iterrows():      # b = holdout's true label level
        cost.append({"ship_level_for": a.label, "holdout_label": b.label,
                     "extra_log_loss_1e4": kl(b.recommended_tau_same_rule, a.recommended_tau_same_rule) * 1e4})
cost = pd.DataFrame(cost)
mix = float(np.mean(lab_df.recommended_tau_same_rule))
lab_df.to_csv(OUT / "q5_label_window_levels.csv", index=False)
cost.pivot(index="ship_level_for", columns="holdout_label", values="extra_log_loss_1e4").to_csv(OUT / "q5_label_window_cost_matrix.csv")

# ------------------------------------------------------------------ 5. walk-forward recommended variant
z = np.load(CACHE / "s3_preds.npz")
testrows = pd.read_parquet(ROOT / "data/splits/walkforward_test_rows.parquet")
pn = pd.read_parquet(ROOT / "data/interim/panel.parquet",
                     columns=["row_id", "id", "y", "asst_tot", "ateco_section", "macro_area", "legal_struct"])
base = pd.DataFrame({"row_id": z["row_id"], "fold": z["fold"]}).merge(testrows, on=["row_id", "fold"]).merge(pn, on="row_id")
VARIANTS = {"CAL-01": ("GB-01|e_last_6m_annualised|M1", "GB-01, prior shift (Saerens eq. 4) to rule e (last 6 months annualised), floor 0.03%, cap 99%"),
            "CAL-02": ("RC-02|e_last_6m_annualised|M1", "RC-02, prior shift to rule e, floor 0.03%, cap 99%"),
            "CAL-03": ("GB-01|b_last_year|M1", "GB-01, prior shift to rule b (last training year rate), floor 0.03%, cap 99%")}
for vid, (key, desc) in VARIANTS.items():
    d = EXP / vid
    d.mkdir(exist_ok=True)
    p = np.clip(z[key], FLOOR, CAP)
    pd.DataFrame({"row_id": base.row_id, "fold": base.fold, "pd": p}).to_parquet(d / "oot_predictions.parquet", index=False)
    json.dump({"exp": vid, "source_predictions": key.split("|")[0], "rule": key.split("|")[1], "method": key.split("|")[2],
               "desc": desc, "floor": FLOOR, "cap": CAP, "script": "experiments/calibration/s3_methods.py + s5_final_level.py",
               "leakage": "targets from training-window labels only (s2); shift = logit(tau) - logit(training rate)"},
              open(d / "config.json", "w"), indent=1)

y = base.y.astype(int).to_numpy()
p_rec = np.clip(z["GB-01|e_last_6m_annualised|M1"], FLOOR, CAP)
p_m0 = z["GB-01|none|M0"]
# floor / cap effect
fc = []
for name, p in (("GB-01 M0", p_m0), ("GB-01 M1-e (unclipped)", z["GB-01|e_last_6m_annualised|M1"])):
    for cap in (0.50, CAP):
        fc.append({"variant": name, "floor": FLOOR, "cap": cap, "min_pd": p.min(), "max_pd": p.max(),
                   "n_below_floor": int((p < FLOOR).sum()), "defaults_below_floor": int(y[p < FLOOR].sum()),
                   "n_above_cap": int((p > cap).sum()), "defaults_above_cap": int(y[p > cap].sum()),
                   "log_loss_unclipped": M.log_loss(y, p), "log_loss_clipped": M.log_loss(y, np.clip(p, FLOOR, cap))})
pd.DataFrame(fc).to_csv(OUT / "floor_cap_effect.csv", index=False)

# subgroup calibration (pooled test years), recommended walk-forward variant vs M0
base["size_q"] = pd.qcut(base.asst_tot.rank(method="first"), 5, labels=["Q1 small", "Q2", "Q3", "Q4", "Q5 large"])
base["seen"] = np.where(base.firm_seen_in_train, "seen firm", "new firm")
sub = []
for gcol in ("seen", "size_q", "ateco_section", "macro_area", "test_year"):
    for g, idx in base.groupby(gcol, observed=True).groups.items():
        ii = base.index.get_indexer(idx)
        if y[ii].sum() < 20:
            continue
        lo, hi = cp(y[ii].sum(), len(ii))
        r0 = M.calibration_slope_intercept(y[ii], p_rec[ii])
        sub.append({"dimension": gcol, "group": str(g), "n": len(ii), "defaults": int(y[ii].sum()),
                    "observed_dr": y[ii].mean(), "dr_ci_low": lo, "dr_ci_high": hi,
                    "mean_pd_M0": p_m0[ii].mean(), "mean_pd_CAL01": p_rec[ii].mean(),
                    "obs_over_pred_CAL01": y[ii].mean() / p_rec[ii].mean(), "cal_slope_CAL01": r0["cal_slope"],
                    "auc_CAL01": M.auc(y[ii], p_rec[ii]), "log_loss_M0": M.log_loss(y[ii], p_m0[ii]),
                    "log_loss_CAL01": M.log_loss(y[ii], p_rec[ii])})
pd.DataFrame(sub).to_csv(OUT / "subgroup_calibration.csv", index=False)

# ------------------------------------------------------------------ 6. rating-grade table (2010 = fold 3, clean)
EDGES = [FLOOR, 0.001, 0.0025, 0.005, 0.0075, 0.01, 0.015, 0.025, 0.05, 0.10, 0.20, CAP + 1e-9]   # pre-declared
LAB = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"]
m10 = (base.test_year == 2010).to_numpy()
g = pd.DataFrame({"y": y[m10], "pd": p_rec[m10]})
g["grade"] = pd.cut(g.pd, [0] + EDGES[1:], labels=LAB, right=False)
gt = g.groupby("grade", observed=False).agg(n=("y", "size"), defaults=("y", "sum"), mean_pd=("pd", "mean")).reset_index()
gt["pd_band"] = [f"[{a:.2%}, {b:.2%})" if i < len(LAB) - 1 else f"[{a:.2%}, {CAP:.0%}]" for i, (a, b) in enumerate(zip([FLOOR] + EDGES[1:-1], EDGES[1:]))]
gt["share_of_firms"] = gt.n / gt.n.sum()
gt["observed_dr"] = gt.defaults / gt.n.where(gt.n > 0)
ci = [cp(k, n) if n > 0 else (np.nan, np.nan) for k, n in zip(gt.defaults, gt.n)]
gt["dr_ci_low"], gt["dr_ci_high"] = [c[0] for c in ci], [c[1] for c in ci]
gt["mean_pd_inside_ci"] = (gt.mean_pd >= gt.dr_ci_low) & (gt.mean_pd <= gt.dr_ci_high)
gt["share_of_defaults"] = gt.defaults / gt.defaults.sum()
gt = gt[["grade", "pd_band", "n", "share_of_firms", "defaults", "share_of_defaults", "mean_pd", "observed_dr",
         "dr_ci_low", "dr_ci_high", "mean_pd_inside_ci"]]
gt.to_csv(OUT / "grade_table_2010.csv", index=False)

# ------------------------------------------------------------------ 7. figures
res = pd.read_csv(OUT / "methods_by_year.csv")
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
yrs = [2008, 2009, 2010]
obs = [R["s6"].loc[t] for t in yrs]
ct = pd.read_csv(OUT / "ct_targets.csv").query("label=='s6'")
for rule, mk in (("a_pooled", "s"), ("b_last_year", "^"), ("c_trend_logit", "v"), ("e_last_6m_annualised", "o")):
    ax[0].plot(yrs, [ct.query("test_year==@t and rule==@rule").target.iloc[0] * 100 for t in yrs], marker=mk, ls="--", label=rule)
ax[0].plot(yrs, [x * 100 for x in obs], "k-", lw=2.5, marker="D", label="observed (test year)")
ax[0].set_title("Central-tendency targets vs observed (start-6 label)")
ax[0].set_ylabel("default rate, %"); ax[0].set_xticks(yrs); ax[0].legend(fontsize=7)
for meth, rule, lbl in (("M0", "none", "GB-01 as submitted"), ("M1", "b_last_year", "GB-01 + rule b"),
                        ("M1", "e_last_6m_annualised", "GB-01 + rule e (CAL-01)"), ("M1", "c_trend_logit", "GB-01 + rule c")):
    s = res.query("model=='GB-01' and method==@meth and rule==@rule and group!='ALL'").sort_values("group")
    ax[1].plot(yrs, s.ratio_obs_to_pred.to_numpy(), marker="o", label=lbl)
ax[1].axhline(1, color="k", lw=1)
ax[1].set_title("Calibration-in-the-large: observed / mean PD"); ax[1].set_xticks(yrs); ax[1].legend(fontsize=7)
fig.tight_layout(); fig.savefig(OUT / "fig/level_by_year.png", dpi=130)

fig, axs = plt.subplots(1, 3, figsize=(13, 4))
for a, t in zip(axs, yrs):
    m = (base.test_year == t).to_numpy()
    for p, lbl, c in ((p_m0, "as submitted (M0)", "tab:gray"), (p_rec, "CAL-01 (M1, rule e)", "tab:blue")):
        d = M.decile_table(y[m], p[m], n_bins=20)
        a.errorbar(d.mean_pd * 100, d.observed_dr * 100,
                   yerr=[(d.observed_dr - d.dr_ci_low) * 100, (d.dr_ci_high - d.observed_dr) * 100],
                   fmt="o", ms=3, capsize=2, color=c, label=lbl)
    lim = [0.03, 30]
    a.plot(lim, lim, "k:", lw=1); a.set_xscale("log"); a.set_yscale("log")
    a.set_title(f"{t}: observed vs predicted, 20 PD bins (95% CP bands)", fontsize=9)
    a.set_xlabel("mean PD in bin, %"); a.set_ylabel("observed default rate, %"); a.legend(fontsize=7)
fig.tight_layout(); fig.savefig(OUT / "fig/binned_calibration_by_year.png", dpi=130)

print(final_rules.round(5).to_string()); print(rw.round(5).to_string())
print(p11.round(5).to_string())
print(pd.DataFrame(lw).round(5).to_string())
print(pd.DataFrame(hl).round(5).to_string())
print(lab_df.round(5).to_string()); print(cost.pivot(index="ship_level_for", columns="holdout_label", values="extra_log_loss_1e4").round(3)); print("mix", mix)
print(pd.DataFrame(fc).to_string())
print(pd.DataFrame(sub).round(4).to_string())
print(gt.round(5).to_string())
