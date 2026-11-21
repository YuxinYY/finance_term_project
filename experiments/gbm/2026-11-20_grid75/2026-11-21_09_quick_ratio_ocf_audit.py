"""Quick ratio + operating cash flow ratio audit (2026-11-21, KN request).

Questions: (1) what values does quick_ratio take and why are the extremes extreme; (2) what shape does the
default rate have before fitting; (3) what do the final unconstrained model (D-160) and the full-monotone
variant mono20 (D-161) do with quick_ratio and operating_cash_flow_ratio (exact TreeSHAP, log-odds).

Data: the 835,016 FY2007-2011 rows the final model was fit on
(experiments/final/2026-11-21_final_fit/final_train_features.parquet), joined to the raw training file on
(id, fs_year) for the ratio components and the sector. In-sample description only; no evaluation metric.
Outputs: experiments/gbm/2026-11-20_grid75/quick_ratio_audit/*.csv, summary.json
"""
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[3]
FIT = REPO / "experiments/final/2026-11-21_final_fit"
OUT = Path(__file__).resolve().parent / "quick_ratio_audit"
OUT.mkdir(exist_ok=True)
RAW = REPO / "data/raw/train-2025.csv"
SEED, N_SHAP = 0, 100_000

feat = pd.read_parquet(FIT / "final_train_features.parquet")
raw = pd.read_csv(RAW, usecols=["id", "fs_year", "ateco_sector", "cash_and_equiv", "AR", "debt_st",
                                "asst_current", "asst_tot"])
d = feat.merge(raw, on=["id", "fs_year"], how="left", validate="1:1")
assert d.asst_tot.notna().all()
summary = {"rows": len(d), "default_rate": float(d.label.mean())}


def deciles(x, k=10):
    return pd.qcut(x.rank(method="first"), k, labels=False) + 1


# ---- 1. quick ratio: raw value (same rule as feature_rules._safe_divide) and distribution
num, den = d.cash_and_equiv + d.AR, d.debt_st
q = num / den.mask(den == 0, den + 1)
summary["qr_feature_equals_raw_share"] = float(np.isclose(d.quick_ratio[q.notna()], q[q.notna()], rtol=1e-4).mean())
summary["qr_missing_raw"] = int(q.isna().sum())
pct = [0, .001, .01, .05, .1, .25, .5, .75, .9, .95, .99, .999, 1]
pd.DataFrame({"percentile": pct, "quick_ratio": q.quantile(pct).values}).to_csv(OUT / "qr_distribution.csv", index=False)
summary["qr_mean"] = float(q.mean())

# ---- 2. value states
st = pd.Series("0-1", index=d.index)
st[q >= 1] = "1-3"; st[q >= 3] = "3-10"; st[q >= 10] = "10-100"; st[q >= 100] = ">=100"
st[q == 0] = "exactly 0"; st[q < 0] = "negative ratio"; st[den == 0] = "debt_st = 0"
st[den < 0] = "debt_st < 0"; st[q.isna()] = "missing component"
order = ["missing component", "debt_st < 0", "debt_st = 0", "negative ratio", "exactly 0", "0-1", "1-3",
         "3-10", "10-100", ">=100"]
g = d.assign(state=st, cash_ar=num).groupby("state").agg(
    n=("label", "size"), default_rate=("label", "mean"), median_debt_st=("debt_st", "median"),
    median_cash_plus_ar=("cash_ar", "median"), median_assets=("asst_tot", "median"))
g["share"] = g.n / len(d)
g.reindex(order).dropna(how="all").to_csv(OUT / "qr_states.csv")
summary["rows_negative_cash"] = int((d.cash_and_equiv < 0).sum())
summary["rows_negative_AR"] = int((d.AR < 0).sum())
summary["rows_negative_debt_st"] = int((d.debt_st < 0).sum())

cols = ["id", "fs_year", "cash_and_equiv", "AR", "debt_st", "asst_tot", "label"]
ext = pd.concat([d.assign(qr=q).nsmallest(5, "qr").assign(end="lowest"),
                 d.assign(qr=q).nlargest(5, "qr").assign(end="highest")])
ext[["end", *cols, "qr"]].to_csv(OUT / "qr_extremes.csv", index=False)

share = d.debt_st / d.asst_tot
rows = []
for name, m in [("top 1%", q >= q.quantile(.99)), ("bottom 1%", q <= q.quantile(.01)), ("all", q.notna())]:
    rows.append({"group": name, "median_debt_st_to_assets": share[m].median(),
                 "median_cash_ar_to_assets": (num / d.asst_tot)[m].median(), "default_rate": d.label[m].mean()})
pd.DataFrame(rows).to_csv(OUT / "qr_tails.csv", index=False)

# ---- 3. univariate default rate by decile, pooled and by year
d["qd"] = deciles(d.quick_ratio)
t = d.groupby("qd").agg(qr_low=("quick_ratio", "min"), qr_high=("quick_ratio", "max"), n=("label", "size"),
                        default_rate=("label", "mean"))
for yr, gy in d.groupby("fs_year"):
    t[f"dr_{yr}"] = gy.groupby("qd").label.mean()
t.to_csv(OUT / "qr_deciles.csv")
summary["qr_decile_dr_spearman_pooled"] = float(spearmanr(t.index, t.default_rate)[0])
summary["qr_decile_dr_spearman_by_year"] = {
    str(yr): float(spearmanr(t.index, t[f"dr_{yr}"])[0]) for yr in sorted(d.fs_year.unique())}
summary["spearman_quick_vs_cur_debt_ratio"] = float(spearmanr(d.quick_ratio, d.cur_debt_ratio)[0])

# ---- 4. sector mix of the deciles (ATECO 68 real estate, 41 building construction)
d["sec"] = d.ateco_sector.astype("Int64").astype(str).str[:2]
d["sector_group"] = d.sec.map({"68": "real estate (68)", "41": "building construction (41)"}).fillna("other")
sg = d.groupby(["qd", "sector_group"]).label.agg(n="size", default_rate="mean").reset_index()
sg.to_csv(OUT / "qr_deciles_by_sector.csv", index=False)
bot = d[d.qd == 1].sector_group.value_counts(normalize=True)
summary["bottom_decile_sector_share"] = bot.to_dict()
summary["all_rows_sector_share"] = d.sector_group.value_counts(normalize=True).to_dict()
summary["sector_default_rate"] = d.groupby("sector_group").label.mean().to_dict()
d["non_quick_share_of_current_assets"] = 1 - num / d.asst_current
summary["median_non_quick_share_of_current_assets"] = d.groupby("sector_group").non_quick_share_of_current_assets.median().to_dict()

# ---- 5. SHAP by decile: final unconstrained (D-160) vs mono20 (D-161)
models = {"free": (FIT / "final_lightgbm.txt", FIT / "final_params.json"),
          "mono20": (FIT / "variants/mono20/lightgbm.txt", FIT / "variants/mono20/params.json")}
s = d.sample(N_SHAP, random_state=SEED).copy()
s["ocf_d"] = deciles(d.operating_cash_flow_ratio).loc[s.index]
for name, (mf, pf) in models.items():
    feats = json.loads(pf.read_text())["feature_columns"]
    c = lgb.Booster(model_file=str(mf)).predict(s[feats], pred_contrib=True)
    for f in ["quick_ratio", "operating_cash_flow_ratio", "operating_cash_flow_to_debt"]:
        s[f"{name}__{f}"] = c[:, feats.index(f)]
    for f in ["quick_ratio", "operating_cash_flow_ratio"]:
        summary[f"spearman_value_vs_shap__{name}__{f}"] = float(spearmanr(s[f], s[f"{name}__{f}"])[0])

qs = s.groupby("qd")[["free__quick_ratio", "mono20__quick_ratio"]].mean()
pd.concat([t[["qr_low", "qr_high", "default_rate"]], qs], axis=1).to_csv(OUT / "qr_shap_deciles.csv")

d["ocf_d"] = deciles(d.operating_cash_flow_ratio)
ot = d.groupby("ocf_d").agg(ocf_low=("operating_cash_flow_ratio", "min"), ocf_high=("operating_cash_flow_ratio", "max"),
                            default_rate=("label", "mean"))
os_ = s.groupby("ocf_d")[["free__operating_cash_flow_ratio", "mono20__operating_cash_flow_ratio",
                          "free__operating_cash_flow_to_debt", "mono20__operating_cash_flow_to_debt"]].mean()
pd.concat([ot, os_], axis=1).to_csv(OUT / "ocf_shap_deciles.csv")
summary["spearman_ocf_ratio_vs_ocf_to_debt"] = float(spearmanr(d.operating_cash_flow_ratio, d.operating_cash_flow_to_debt)[0])

(OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
print(json.dumps(summary, indent=2, default=float))
