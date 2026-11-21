"""Profile of the submission model mono20 (D-161/D-163): global SHAP, family sums, PD distribution, in-sample
grade table, example reason codes, tree statistics, and agreement with the unconstrained D-160 model.

Data: all 835,016 FY2007-2011 training rows (in-sample: describes the model, not out-of-sample quality).
Exact TreeSHAP via LightGBM pred_contrib (log-odds of the raw model; the level step adds one constant).
Outputs: reports/final_model/2026-11-21_mono20_*.csv, 2026-11-21_mono20_summary.json
"""
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
OUT = REPO / "reports/final_model"
P = json.loads((HERE / "variants/mono20/params.json").read_text())
feats = P["feature_columns"]
cal = P["calibration"]
booster = lgb.Booster(model_file=str(HERE / "variants/mono20/lightgbm.txt"))
free = lgb.Booster(model_file=str(HERE / "final_lightgbm.txt"))
free_cal = json.loads((HERE / "final_params.json").read_text())["calibration"]
sign = pd.read_csv(REPO / "experiments/gbm/2026-11-20_grid75/sign_table.csv").set_index("feature")

df = pd.read_parquet(HERE / "final_train_features.parquet")
y = df.label.to_numpy()


def to_pd(raw, c):
    lo = logit(np.clip(raw, c["epsilon"], 1 - c["epsilon"]))
    return np.clip(expit(c["anchor_a"] + c["slope_b"] * lo + c["delta"]), c["floor"], c["cap"])


raw = booster.predict(df[feats])
pdv = to_pd(raw, cal)
pd_free = to_pd(free.predict(df[feats]), free_cal)
summary = {"rows": len(df), "defaults": int(y.sum()), "train_rate": float(y.mean()),
           "trees": booster.num_trees(), "mean_raw_pd": float(raw.mean()), "mean_final_pd": float(pdv.mean()),
           "insample_auc": float(roc_auc_score(y, raw)), "pd_percentiles": {}}
for q in [0, .01, .1, .25, .5, .75, .9, .99, .999, 1]:
    summary["pd_percentiles"][str(q)] = float(np.quantile(pdv, q))
summary["share_at_floor"] = float((pdv <= cal["floor"] + 1e-12).mean())
summary["spearman_pd_mono20_vs_free"] = float(spearmanr(pdv, pd_free)[0])
summary["mean_abs_pd_diff_vs_free"] = float(np.abs(pdv - pd_free).mean())
summary["share_rows_pd_ratio_within_20pct_of_free"] = float((np.abs(pdv / pd_free - 1) <= 0.2).mean())

# tree statistics
td = booster.trees_to_dataframe()
leaves = td[td.left_child.isna()].groupby("tree_index").size()
splits = td[td.split_feature.notna()].groupby("split_feature").size()
summary["leaves_per_tree_mean"] = float(leaves.mean())
summary["features_used_in_splits"] = int(splits.size)
summary["features_never_split"] = sorted(set(booster.feature_name()) - set(splits.index))

# exact SHAP on all rows
contrib = booster.predict(df[feats], pred_contrib=True)
sh = pd.DataFrame(contrib[:, :-1], columns=feats)
summary["shap_base_value_logodds"] = float(contrib[0, -1])
g = pd.DataFrame({"feature": feats, "mean_abs_shap": sh.abs().mean().values})
g["rank"] = g.mean_abs_shap.rank(ascending=False, method="first").astype(int)
g["expected_sign"] = [int(sign.loc[f, "candidate"]) if f in sign.index else 0 for f in feats]
g["constrained"] = [int(sign.loc[f, "constraint"]) if f in sign.index else 0 for f in feats]
g["value_vs_shap_spearman"] = [spearmanr(df[f], sh[f])[0] if sh[f].std() > 0 else np.nan for f in feats]
g["splits"] = [int(splits.get(booster.feature_name()[i], 0)) for i in range(len(feats))]
g = g.sort_values("rank")
g.to_csv(OUT / "2026-11-21_mono20_shap_global.csv", index=False)

fam = {
    "leverage / capital structure": ["equity_ratio", "debt_to_eqty", "st_debt_share", "lt_debt_to_capital", "bank_debt_conc",
                                     "cur_asst_ratio", "tangible_asset_ratio", "goodwill_ratio", "neg_bs_flag"],
    "liquidity / activity": ["cash_ratio", "AR_ratio", "payables_ratio", "wc_ratio", "cur_debt_ratio", "quick_ratio",
                             "asset_turnover"],
    "profitability": ["roa", "roe", "op_margin", "net_prof_margin", "profit", "ebitda", "COGS", "loss_flag",
                      "roe_negative", "neg_pl_flag"],
    "coverage / cash flow": ["exp_financing", "int_coverage", "financing_expense_ratio", "cash_to_financing_exp_ratio",
                             "ebitda_to_debt", "operating_cash_flow_margin", "cash_interest_coverage",
                             "operating_cash_flow_ratio", "operating_cash_flow_to_debt"],
    "size": ["size"],
    "industry": ["ateco_sector_te"],
    "legal form": [f for f in feats if f.startswith("legal_")],
    "region": [f for f in feats if f.startswith("city_")],
    "data-quality / zero flags": [f for f in feats if f.startswith("zero_")] + ["debt_detail_missing"],
}
rows = []
for name, members in fam.items():
    m = [f for f in members if f in feats]
    s = sh[m].sum(axis=1)
    rows.append({"family": name, "n_features": len(m), "mean_abs_family_shap": float(s.abs().mean()),
                 "share_of_total_abs_shap": float(sh[m].abs().sum().sum() / sh.abs().sum().sum())})
pd.DataFrame(rows).sort_values("mean_abs_family_shap", ascending=False).to_csv(
    OUT / "2026-11-21_mono20_shap_family.csv", index=False)

# in-sample grade table (PD deciles) and per-year level
d = pd.DataFrame({"fs_year": df.fs_year, "y": y, "pd": pdv})
d["pd_decile"] = pd.qcut(d.pd.rank(method="first"), 10, labels=False) + 1
gt = d.groupby("pd_decile").agg(n=("y", "size"), pd_low=("pd", "min"), pd_high=("pd", "max"),
                                mean_pd=("pd", "mean"), observed_dr=("y", "mean"), defaults=("y", "sum"))
gt["share_of_defaults"] = gt.defaults / gt.defaults.sum()
gt.to_csv(OUT / "2026-11-21_mono20_grade_table_insample.csv")
d.groupby("fs_year").agg(n=("y", "size"), observed_dr=("y", "mean"), mean_pd=("pd", "mean")).to_csv(
    OUT / "2026-11-21_mono20_level_by_year_insample.csv")

# example reason codes: firms at the 10th, 50th, 90th, 99th PD percentile
ex = []
for q in [.10, .50, .90, .99]:
    i = int(np.argmin(np.abs(pdv - np.quantile(pdv, q))))
    top = sh.iloc[i].reindex(sh.iloc[i].abs().sort_values(ascending=False).index)[:5]
    for r, (f, v) in enumerate(top.items(), 1):
        ex.append({"pd_percentile": q, "pd": pdv[i], "label": int(y[i]), "rank": r, "feature": f,
                   "feature_value": float(df[f].iloc[i]), "shap_logodds": float(v),
                   "direction": "raises PD" if v > 0 else "lowers PD"})
pd.DataFrame(ex).to_csv(OUT / "2026-11-21_mono20_firm_examples.csv", index=False)

(OUT / "2026-11-21_mono20_summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
print(json.dumps(summary, indent=2, default=float))
print(g.head(25).round(4).to_string(index=False))
print(pd.read_csv(OUT / "2026-11-21_mono20_shap_family.csv").round(4).to_string(index=False))
print(gt.round(5).to_string())
print(pd.read_csv(OUT / "2026-11-21_mono20_level_by_year_insample.csv").round(5).to_string(index=False))
print(pd.DataFrame(ex).round(4).to_string(index=False))
