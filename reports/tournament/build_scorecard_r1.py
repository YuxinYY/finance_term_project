"""Render the numeric tables of scorecard_r1.md from the CSVs written by score_r1.py.

    python reports/tournament/build_scorecard_r1.py   -> reports/tournament/r1_tables.md

The prose in scorecard_r1.md quotes these tables; no metric is typed by hand.
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
met = pd.read_csv(OUT / "r1_metrics_long.csv")
met["group"] = met["group"].astype(str)
lb = pd.read_csv(OUT / "r1_leaderboard_ci.csv")
pb = pd.read_csv(OUT / "r1_paired_bootstrap.csv")
pb["subset"] = pb["subset"].astype(str)
fv = pd.read_csv(OUT / "r1_face_validity_long.csv")
gbpd = pd.read_csv(OUT / "r1_face_validity_gbm_pd.csv")
L = []


def f(x, d=4):
    return "" if pd.isna(x) else f"{x:.{d}f}"


def cis(p, lo, hi, d=4, sign=False):
    s = "+" if sign else ""
    return f"{p:{s}.{d}f} [{lo:{s}.{d}f}, {hi:{s}.{d}f}]"


# --- leaderboard
allm = met[met.group == "ALL"].set_index("exp")
lbi = lb.set_index("exp")
rows = []
for e in lbi.index:
    r, m = lbi.loc[e], allm.loc[e]
    rows.append({"exp": e, "AUC [95% CI]": cis(r.auc_ALL, r.auc_ALL_lo, r.auc_ALL_hi),
                 "AR": f(m.ar, 3), "log loss [95% CI]": cis(r.ll_ALL, r.ll_ALL_lo, r.ll_ALL_hi, 5),
                 "log loss L1 (leak-free level shift)": f(m.log_loss_L1, 5),
                 "log loss L2 (oracle, diagnostic)": f(m.log_loss_L2_oracle, 5),
                 "Brier": f(m.brier, 5), "mean PD": f(m.mean_pd), "obs DR": f(m.observed_dr),
                 "obs/pred": f(m.ratio_obs_to_pred, 2), "cal. intercept": f(m.cal_intercept, 3),
                 "cal. slope": f(m.cal_slope, 3)})
t = pd.DataFrame(rows).sort_values("AUC [95% CI]", ascending=False)
L += ["## T1 Leaderboard (pooled 2008-2010, 506,159 rows; CIs = firm-clustered bootstrap percentiles, 200 draws)", "",
      t.to_markdown(index=False), ""]

# --- per year
rows = []
for e in lbi.index:
    r = {"exp": e}
    for g in ("2008", "2009", "2010"):
        m = met[(met.exp == e) & (met.group == g)].iloc[0]
        r[f"AUC {g}"] = f(m.auc)
        r[f"LL {g}"] = f(m.log_loss, 5)
        r[f"obs/pred {g}"] = f(m.ratio_obs_to_pred, 2)
        r[f"slope {g}"] = f(m.cal_slope, 2)
    rows.append(r)
L += ["## T2 Per test year (AUC, log loss, obs/pred level ratio, calibration slope)", "",
      pd.DataFrame(rows).to_markdown(index=False), ""]
yy = met[(met.exp == lbi.index[0]) & met.group.isin(["2008", "2009", "2010", "seen", "new"])][["group", "n", "defaults", "observed_dr"]]
L += ["Test-set sizes (identical for every model):", "", yy.to_markdown(index=False), ""]

# --- seen vs new
rows = []
for e in lbi.index:
    s = met[(met.exp == e) & (met.group == "seen")].iloc[0]
    n = met[(met.exp == e) & (met.group == "new")].iloc[0]
    r = lbi.loc[e]
    rows.append({"exp": e, "AUC seen": f(s.auc), "AUC new": f(n.auc),
                 "gap seen-new [95% CI]": cis(r.seen_minus_new_auc, r.seen_minus_new_lo, r.seen_minus_new_hi),
                 "LL seen": f(s.log_loss, 5), "LL new": f(n.log_loss, 5),
                 "slope seen": f(s.cal_slope, 2), "slope new": f(n.cal_slope, 2)})
L += ["## T3 Seen-firm vs new-firm rows (pooled)", "", pd.DataFrame(rows).to_markdown(index=False), ""]

# --- paired comparisons
def ptable(sub, title):
    rows = []
    for _, r in sub.iterrows():
        rows.append({"new vs old": f"{r.new} vs {r.old}", "subset": r.subset,
                     "dAUC [95% CI]": cis(r.d_auc, r.d_auc_lo, r.d_auc_hi, 4, True),
                     "dLogLoss [95% CI]": cis(r.d_log_loss, r.d_log_loss_lo, r.d_log_loss_hi, 5, True),
                     "dLogLoss L1 [95% CI]": cis(r.d_log_loss_L1, r.d_log_loss_L1_lo, r.d_log_loss_L1_hi, 5, True),
                     "AUC CI excl. 0": "new better" if r.d_auc_lo > 0 else ("new worse" if r.d_auc_hi < 0 else "tie"),
                     "LL CI excl. 0": "new better" if r.d_log_loss_hi < 0 else ("new worse" if r.d_log_loss_lo > 0 else "tie")})
    return [title, "", pd.DataFrame(rows).to_markdown(index=False), ""]


L += ptable(pb[(pb.old == "SC-03") & (pb.subset == "ALL")].drop_duplicates(["new", "old", "subset"]),
            "## T4 Paired firm-clustered bootstrap vs incumbent SC-03 (pooled; negative dLogLoss favours new)")
L += ptable(pb[(pb.old == "SC-03") & pb.new.isin(["GB-01", "RC-03", "RC-01", "GB-03"]) & (pb.subset != "ALL")].drop_duplicates(["new", "old", "subset"]),
            "## T5 Same, by test year and seen/new firms (fragility check)")
key = pb[~((pb.old == "SC-03") & (pb.new != "SC-02"))].drop_duplicates(["new", "old", "subset"])
L += ptable(key[key.subset == "ALL"], "## T6 Key pairs (pooled)")
L += ptable(key[key.subset.isin(["2008", "2009", "2010"]) & key.new.isin(["RC-03", "GB-01", "GB-02"]) & key.old.isin(["RC-01", "RC-03", "GB-01"])],
            "## T7 Key pairs by test year")

# --- face validity matrix
feats = [x for x in gbpd.feature.unique()]
mat = {}
for e, g in fv.groupby("exp"):
    row = {}
    for ft, gg in g.groupby("feature"):
        if gg.violates.any():
            row[ft] = "VIOL(f" + ",".join(str(int(k)) for k in gg[gg.violates].fold) + ")"
        elif "partial" in gg and gg["partial"].fillna(False).astype(bool).any():
            row[ft] = "partial"
        else:
            row[ft] = "ok"
    mat[e] = row
M_ = pd.DataFrame(mat).T.reindex(columns=feats).fillna("-")
M_.insert(0, "violations", [int(fv[(fv.exp == e) & fv.violates].feature.nunique()) for e in M_.index])
order = ["SC-01", "SC-02", "SC-03", "SC-04", "RC-01", "RC-02", "RC-03", "GB-01", "GB-02", "GB-03", "GB-04", "GB-05", "GB-06"]
M_ = M_.reindex(order)
L += ["## T8 Face-validity matrix (features with an expected sign; '-' = not in the model)", "",
      "Linear models: coefficient sign per fold (all 3 folds). SC-03: beta on risk-oriented WoE > 0 and WoE monotone. "
      "RiskCalc: probit coefficient on Phi^-1(T_i) > 0 per fold and T_i monotone (all folds). GBM: partial dependence of the "
      "saved fold-3 model on 20,000 fold-3 training rows, 25 quantile grid points; VIOL = more than half of the curve's "
      "movement is against the expected sign; partial = 10-50%. 'violations' counts features.", "",
      M_.to_markdown(), ""]
gsub = gbpd[gbpd.exp.isin(["GB-01", "GB-02", "GB-06"]) & (gbpd.violation_share > 0)].copy()
gsub["violation_share"] = gsub.violation_share.round(3)
gsub["range"] = gsub["range"].round(3)
L += ["## T9 GBM partial-dependence violation shares > 0 (fold-3 models)", "",
      gsub[["exp", "feature", "expected", "constrained", "range", "violation_share"]].to_markdown(index=False), ""]
coefv = fv[fv.violates & ~fv.exp.str.startswith("GB")][["exp", "fold", "feature", "evidence", "value", "z"]].copy()
coefv["value"] = coefv.value.round(4)
coefv["z"] = coefv.z.round(2)
L += ["## T10 Coefficient sign violations (linear / probit models)", "", coefv.to_markdown(index=False), ""]

# --- misc
for name, title in (("r1_contract.csv", "## T11 Contract check"), ("r1_d011_label_check.csv", "## T12 D-011 label check (panel y vs make_labels, start 6, data_end 2013-05-31)"),
                    ("r1_bootstrap_equivalence.csv", "## T13 Fast bootstrap vs metrics.paired_bootstrap (GB-01 vs SC-03, 15,000-firm subsample, 40 draws, seed 0)"),
                    ("r1_level_shift_L1.csv", "## T14 L1 level shift per fold (logit; training labels only)"),
                    ("r1_repro_check.csv", "## T15 Reproduction check: fold 3 refitted by the referee from current code"),
                    ("r1_tangible_ratio_check.csv", "## T16 tangible_ratio: univariate AUC (PD falls with ratio) within ATECO group, fold-3 training rows")):
    p = OUT / name
    if p.exists():
        d = pd.read_csv(p)
        L += [title, "", d.to_markdown(index=False, floatfmt=".6g"), ""]
(OUT / "r1_tables.md").write_text("\n".join(L))
print("\n".join(L))

# --- fill the template
txt = "\n".join(L)
secs, cur = {}, None
for line in txt.splitlines():
    if line.startswith("## T"):
        cur = line.split()[1]
        secs[cur] = []
    if cur:
        secs[cur].append(line)
blk = lambda ks: "\n".join("\n".join(secs[k]).replace("## T", "### T") for k in ks if k in secs)
tpl = (OUT / "scorecard_r1_template.md").read_text()
tpl = tpl.replace("{{TABLES_A}}", blk(["T1", "T2", "T3", "T4", "T5", "T6", "T7"]))
tpl = tpl.replace("{{TABLES_FV}}", blk(["T8", "T9", "T10", "T16"]))
tpl = tpl.replace("{{TABLES_B}}", "## Appendix: checks\n\n" + blk(["T11", "T12", "T13", "T14", "T15"]))
(OUT / "scorecard_r1.md").write_text(tpl)
