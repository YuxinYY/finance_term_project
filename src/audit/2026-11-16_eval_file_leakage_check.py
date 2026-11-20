"""Check whether Chloe's round-1 validation/test feature files were fit on their own year (audit finding A1).

Writes reports/data/2026-11-16_audit/r5_eval_file_leakage.csv.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "harness/2026-11-16_turquoise_submission"))
import feature_rules  # noqa: E402
import missing_rules  # noqa: E402

CH = Path.home() / "Downloads/non-aggregated feature data sets (80 features version 11.15)/round 1"
raw = pd.read_csv(ROOT / "data/raw/train-2025.csv", low_memory=False)
pan = pd.read_parquet(ROOT / "data/interim/2026-11-08_panel_s5.parquet", columns=["id", "fs_year", "y_s5"])
raw = raw.merge(pan, on=["id", "fs_year"])
lab = raw[raw.y_s5.notna()]
cfg = SimpleNamespace(id_column="id", year_column="fs_year")
lookup = missing_rules._ateco_industry_lookup()
rows = []
for fname, yr in [("features_validation.csv", 2008), ("features_test.csv", 2009)]:
    ch = pd.read_csv(CH / fname).set_index("id")
    target = raw[raw.fs_year == yr]
    rec = {"file": fname, "fs_year": yr}
    for tag, years in [("train_FY2007", [2007]), ("own_year", [yr])]:
        med = missing_rules.sector_medians(lab[lab.fs_year.isin(years)].drop(columns=["y_s5"]))
        filled = missing_rules.fill_missing(target.drop(columns=["y_s5"]), med, use_firm_history=True).assign(neg_bs_flag=0)  # flag not needed here
        f = feature_rules.make_period_features(filled, cfg).set_index("id")
        miss = target.set_index("id")[["debt_bank_st", "debt_bank_lt", "AP_st"]].isna().any(axis=1)
        idx = miss[miss].index.intersection(ch.index)
        rec[f"imputed_rows_matching_medians_{tag}"] = float(np.isclose(f.loc[idx, "bank_debt_conc"], ch.loc[idx, "bank_debt_conc"], rtol=1e-9).mean())
        rec["imputed_rows_compared"] = len(idx)
    ind = missing_rules._ateco_industry_group(target.ateco_sector, lookup)
    t = pd.DataFrame({"id": target.id.values, "ind": ind.values, "y": target.y_s5.values}).merge(ch[["ateco_sector_te"]], left_on="id", right_index=True)
    trn = raw[raw.fs_year == 2007]
    dr_train = trn.y_s5.groupby(missing_rules._ateco_industry_group(trn.ateco_sector, lookup).values).mean()
    j = pd.concat([t.groupby("ind").ateco_sector_te.mean(), dr_train, t.groupby("ind").y.mean()], axis=1, keys=["te", "dr_train", "dr_own"]).dropna()
    rec.update({"te_distinct_per_industry_median": float(t.groupby("ind").ateco_sector_te.nunique().median()),
                "corr_te_trainyear_industry_dr": j.te.corr(j.dr_train), "corr_te_ownyear_industry_dr": j.te.corr(j.dr_own),
                "mean_te": t.ateco_sector_te.mean(), "own_year_default_rate": t.y.mean(), "train_year_default_rate": trn.y_s5.mean()})
    rows.append(rec)
out = pd.DataFrame(rows)
out.to_csv(ROOT / "reports/data/2026-11-16_audit/r5_eval_file_leakage.csv", index=False)
print(out.T.to_string())
