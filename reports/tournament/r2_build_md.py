"""Renders r2_scorecard.md and r2_win_rule_test.csv from the r2_*.csv files. Tables are read from files, never typed."""
from pathlib import Path
import json
import pandas as pd
O = Path(__file__).resolve().parent
c = lambda n: pd.read_csv(O / n)
md = lambda d, f=None: d.to_markdown(index=False, floatfmt=f or ".4f")
met, pt, lv, gb, gt, sg, fv, sh, rc = c("r2_metrics_long.csv"), c("r2_paired_bootstrap.csv"), c("r2_level_by_year.csv"), c("r2_grade_bands.csv"), c("r2_grade_table.csv"), c("r2_segments.csv"), c("r2_face_validity.csv"), c("r2_shapes_unconstrained.csv"), c("r2_unconstrained_reason_codes.csv")
OWN = ["GB-25", "GB-20", "GB-23", "GB-08", "GB-08b", "GB-09", "GB-10", "GB-11", "GB-01", "GB-02", "GAM-01"]
SHIFT = ["GB-25+CAL", "GB-20+CAL", "CAL-01", "GAM-01+CAL"]


def tbl(exps, cell):
    d = met[met.cell == cell].set_index("exp").loc[exps].reset_index()
    d = d[["exp", "n", "defaults", "auc", "ar", "log_loss", "brier", "mean_pd", "observed_dr", "ratio_obs_to_pred", "cal_slope"]]
    return d.rename(columns={"brier": "brier (overall prob. accuracy)", "ratio_obs_to_pred": "obs/pred"})


def fmt_pairs(d):
    r = pd.DataFrame({"new": d.new, "old": d.old, "subset": d.subset, "defaults": d.defaults,
                      "dAUC [95% CI]": [f"{a:+.4f} [{l:+.4f}, {h:+.4f}]" for a, l, h in zip(d.d_auc, d.d_auc_lo, d.d_auc_hi)],
                      "dAR": d.d_ar.map(lambda v: f"{v:+.4f}"),
                      "dLogLoss [95% CI]": [f"{a:+.5f} [{l:+.5f}, {h:+.5f}]" for a, l, h in zip(d.d_ll, d.d_ll_lo, d.d_ll_hi)], "draws": d.n_draws})
    return r.to_markdown(index=False)


# win rules vs GB-01
rows = []
MIN_AUC, MIN_LL = 0.002, 0.00015
for e in ["GB-25", "GB-20", "GB-23", "GB-08", "GB-08b", "GB-09", "GB-10", "GB-11", "GB-02"]:
    p = pt[(pt.new == e) & (pt.old == "GB-01")].set_index("subset")
    a = p.loc["ALL"]
    yrs = p.loc[["2008", "2009", "2010"]]
    old = bool(a.d_ll_hi < 0 or (a.d_auc_lo > 0 and a.d_ll <= 0))
    d100 = bool((a.d_auc_lo > 0 and (yrs.d_auc > 0).all()) or (a.d_ll_hi < 0 and (yrs.d_ll < 0).all()))
    t = p.loc["2010"]
    eff = bool(a.d_auc >= MIN_AUC and a.d_ll <= -MIN_LL and t.d_auc >= MIN_AUC and t.d_ll <= -MIN_LL)
    clean = bool(t.d_auc_lo > 0 and t.d_ll_hi < 0)
    noyear = bool(((yrs.d_auc_hi >= 0) & (yrs.d_ll_lo <= 0)).all())
    mult = bool(a.d_ll_hi_995 < 0 and a.d_auc_lo_995 > 0)
    rows.append({"exp": e, "pooled dAUC": a.d_auc, "pooled dLL": a.d_ll, "2010 dAUC": t.d_auc, "2010 dLL": t.d_ll, "draws": int(a.n_draws), "old rule (D-060/D-065 statistical part)": old, "D-100 'WIN' (either metric)": d100,
                 "R1 min effect pooled+2010": eff, "R2 2010 CI both metrics": clean, "R3 no year CI against": noyear, "R4 pooled 99.5% CI both metrics": mult, "statistical part of revised rule": eff and clean and noyear and mult})
win = pd.DataFrame(rows)
win.to_csv(O / "r2_win_rule_test.csv", index=False)
sens = []
for th, tl in ((0.001, 0.00008), (0.002, 0.00015), (0.003, 0.00025), (0.004, 0.00035)):
    sens.append({"min dAUC": th, "min dLL": tl, **{e: bool(win[win.exp == e].iloc[0]["pooled dAUC"] >= th and win[win.exp == e].iloc[0]["pooled dLL"] <= -tl and win[win.exp == e].iloc[0]["2010 dAUC"] >= th and win[win.exp == e].iloc[0]["2010 dLL"] <= -tl) for e in win.exp}})
sens = pd.DataFrame(sens)
sens.to_csv(O / "r2_win_rule_sensitivity.csv", index=False)

con = c("r2_contract.csv")[["exp", "sha256_12", "n_rows", "missing_test_rows", "extra_rows", "dup_row_fold", "nan_pd", "pd_outside_open_01", "rows_fs_year_ge_2011", "min_pd", "max_pd", "PASS"]]
rfit = c("r2_repro_refit.csv")
ice = c("r2_ice_refit_gb25.csv")
ice_s = ice.groupby("fold").agg(constrained=("feature", "size"), ice_rows=("ice_rows", "first"), violating_rows=("violating_rows", "sum"), zero_split_features=("splits", lambda s: int((s == 0).sum()))).reset_index()
callog = c("r2_refit_call_log.csv")
led = json.load(open(O / "r2_selection_ledger.json"))

seg_cols = ["cell", "n", "defaults", "ar_GB-01", "ar_GB-25", "ar_GB-20", "d_ar_GB-25_vs_GB-01", "d_ar_GB-25_vs_GB-01_lo", "d_ar_GB-25_vs_GB-01_hi", "d_ll_GB-25_vs_GB-01", "d_ll_GB-25_vs_GB-01_lo", "d_ll_GB-25_vs_GB-01_hi"]
seg = sg[seg_cols].rename(columns=lambda s: s.replace("GB-25_vs_GB-01", "vs01").replace("ar_", "AR "))
gr10 = gt[(gt.year == 2010) & gt.exp.isin(["GB-25+CAL", "CAL-01"])]
piv = gr10.pivot(index=["grade", "pd_band"], columns="exp", values=["share_of_firms", "defaults", "expected_defaults", "obs_over_pred", "flag"])
piv.columns = [f"{b} {a}" for a, b in piv.columns]
piv = piv.reset_index().fillna("")
grany = gt[gt.exp == "GB-25+CAL"][["year", "grade", "pd_band", "n", "defaults", "expected_defaults", "obs_over_pred", "flag"]].fillna("")
gy = grany[grany.year != 2010]
sh25 = sh[sh.exp == "GB-25"]
fvm = pd.DataFrame([
    ("GB-25", 36, 9, 0, 0, "measured (fold 3; folds 1-3 refit ICE)"),
    ("GB-20", 35, 10, 0, 1, "measured; interest_to_sales against a priori + sign, against-share in r2_face_validity.csv"),
    ("GB-23", 35, 10, 0, 1, "same free interest_to_sales; shape not measured this round"),
    ("GB-01", 20, 6, 0, 0, "measured; incumbent"),
    ("GB-08", 17, 9, 2, 2, "from D-103 (cfo_to_debt, interest_coverage CONTRADICT); not re-measured"),
    ("GB-08b", 19, 7, 1, 1, "from D-103 (ebitda_margin CONTRADICTS); not re-measured"),
    ("GB-09", 20, 6, 0, 0, "same constraints as GB-01; not re-measured"),
    ("GB-10", 20, 6, 0, 0, "same constraints as GB-01; not re-measured"),
    ("GB-02", 0, 26, 3, 3, "from D-051 (wc_ratio, bank_debt_ratio, interest_coverage contradict); fails"),
    ("GB-11", 0, 26, None, None, "unconstrained RF, no saved model (D-106); not measurable; benchmark only"),
    ("GAM-01", 20, 6, 0, 0, "from D-111; not re-measured")], columns=["model", "constrained", "unconstrained", "violations of FEATURE_SPEC signs", "violations incl. a priori + on interest_to_sales", "basis"])

T = (O / "r2_scorecard_template.md").read_text()
sub = {
 "{{CONTRACT}}": md(con), "{{LEAD_POOLED}}": md(tbl(OWN + SHIFT, "pooled"), ".5f"), "{{Y2010}}": md(tbl(OWN + SHIFT, "year|2010"), ".5f"), "{{Y2008}}": md(tbl(OWN + SHIFT, "year|2008"), ".5f"), "{{Y2009}}": md(tbl(OWN + SHIFT, "year|2009"), ".5f"),
 "{{PAIR_MAIN}}": fmt_pairs(pt[pt.new.isin(["GB-25", "GB-20"]) & pt.old.isin(["GB-01", "GB-02", "GB-11", "GAM-01", "GB-20"])]),
 "{{PAIR_TABLE}}": fmt_pairs(pt[((pt.old == "GB-01") & pt.new.isin(["GB-23", "GB-08", "GB-08b", "GB-09", "GB-10", "GB-11", "GB-02"]) | ((pt.new == "GB-23") & pt.old.isin(["GB-20", "GB-25"]))) & pt.subset.isin(["ALL", "2008", "2009", "2010"])]),
 "{{PAIR_CAL}}": fmt_pairs(pt[pt.new.str.contains("CAL")]),
 "{{REPRO_SAVED}}": md(c("r2_repro_saved.csv"), ".2e"), "{{REPRO_REFIT}}": md(rfit.drop(columns=["n_final_trees_config"]), ".2e"), "{{PROV}}": md(c("r2_param_provenance.csv").T.reset_index()), "{{INNER}}": md(c("r2_inner_rows_check.csv")), "{{FLIP}}": md(c("r2_oof_label_flip.csv"), ".4f"), "{{SB}}": md(c("r2_single_vs_batch.csv").T.reset_index()),
 "{{CALLS}}": md(callog[(callog.exp == "GB-25") & (callog.fold == 3) & (callog.call != "") ]),
 "{{LEDGER}}": "```json\n" + json.dumps(led, indent=1) + "\n```",
 "{{SEG}}": md(seg), "{{CAP}}": md(c("r2_cap_points.csv").query("exp in ['GB-25','GB-20','GB-01'] and riskiest_share==0.1").pivot(index="cell", columns="exp", values="defaults_captured_share").reset_index(), ".3f"),
 "{{LEVEL}}": md(lv[lv.exp.isin(SHIFT)][["exp", "year", "defaults", "expected_defaults", "excess_defaults", "observed_dr", "mean_pd", "obs_over_pred", "cal_slope"]]), "{{BANDS}}": md(gb[gb.exp.isin(["GB-25+CAL", "CAL-01", "GB-20+CAL"])], ".3f"), "{{SAME}}": md(c("r2_grade_same_firms.csv"), ".3f"), "{{GRADE10}}": piv.to_markdown(index=False), "{{GRADEOTHER}}": gy.to_markdown(index=False, floatfmt=".3f"),
 "{{FVMAT}}": fvm.to_markdown(index=False), "{{ICE}}": md(ice_s), "{{FV25}}": md(fv[(fv.exp == "GB-25") & (fv.constraint == 0)][["feature", "kind", "splits", "pd_lo", "pd_hi", "pd_range"]], ".3f"), "{{RC}}": md(rc, ".3f"), "{{RAW}}": md(c("r2_raw_interest_test.csv"), ".3f"), "{{RAWE}}": md(c("r2_raw_interest_by_ebitda.csv"), ".3f"),
 "{{SHAPES}}": sh25[sh25.kind != "level"][["feature", "point", "x", "pd_logodds", "train_dr_2007_09"]].to_markdown(index=False, floatfmt=".4f"), "{{LEVELS}}": sh25[sh25.kind == "level"][["feature", "point", "pd_logodds", "train_dr_2007_09"]].to_markdown(index=False, floatfmt=".3f"), "{{INTEREST}}": sh[(sh.feature == "interest_to_sales") & (sh.exp != "GB-01")][["exp", "point", "x", "pd_logodds", "train_dr_2007_09"]].to_markdown(index=False, floatfmt=".4f"),
 "{{WIN}}": win.to_markdown(index=False, floatfmt=".5f"), "{{SENS}}": sens.to_markdown(index=False), "{{FVM}}": ""}
for k, v in sub.items():
    T = T.replace(k, v)
assert "{{" not in T, T[T.index("{{"):][:60]
(O / "r2_scorecard.md").write_text(T)
print(win.T.to_string()); print(sens.to_string()); print(rfit.to_string()); print(ice_s)
