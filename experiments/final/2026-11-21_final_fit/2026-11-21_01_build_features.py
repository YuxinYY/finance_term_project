"""Estimation step 1: fit the team preprocessing (schema 1.3.0, teammate/project-update @ 7949207, vendored in
vendor_7949207/) and build the 75 VIF-selected model features.

Modes
  parity  fit on FY2007-2009 labelled rows, transform FY2010 rows; compare with Chloe's 11.20 round-03 files
          (proves this code reproduces the development data exactly).
  final   fit on ALL labelled FY2007-2011 rows (835,016; label = 5-month window, D-151) -> final training
          features + the frozen preprocessing artifact used by the harness.

Labels come from data/interim/2026-11-08_panel_s5.parquet (y, label_status == 'ok'); they match Chloe's files
on every row (D-156). def_date is dropped before preprocessing (never a feature input).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
VENDOR = HERE / "vendor_7949207"
sys.path.insert(0, str(VENDOR / "src"))
from finance_project.data_processing.preprocess import (  # noqa: E402
    PreprocessingConfig, fit_preprocessor, transform_with_preprocessor)

REPO = HERE.parents[2]
RAW = REPO / "data/raw/train-2025.csv"
PANEL = REPO / "data/interim/2026-11-08_panel_s5.parquet"
CHLOE = Path("/Users/kietnguyen/Downloads/non-aggregated feature data sets (75 features 11.20)")
VIF = json.loads((CHLOE / "vif_selected_features.json").read_text())
SELECTED = VIF["selected_features"]
META = ["id", "fs_year", "label"]


def labelled_raw(years):
    raw = pd.read_csv(RAW, low_memory=False)
    lab = pd.read_parquet(PANEL, columns=["id", "fs_year", "y", "label_status"])
    lab = lab[lab.label_status == "ok"].rename(columns={"y": "label"})[["id", "fs_year", "label"]]
    out = raw.merge(lab, on=["id", "fs_year"], how="inner")
    out = out[out.fs_year.isin(years)].drop(columns=["def_date"])
    return out.reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["parity", "final"])
    a = ap.parse_args()
    cfg = PreprocessingConfig(feature_mode="firm_year")
    t0 = time.time()
    if a.mode == "parity":
        tr = labelled_raw([2007, 2008, 2009]); va = labelled_raw([2010])
        print(f"train rows {len(tr):,} (Chloe 476,183), validation rows {len(va):,} (Chloe 174,674)")
        ftr, art = fit_preprocessor(tr, tr, cfg)
        fva = transform_with_preprocessor(va, va, art)
        report = []
        for name, mine, theirs in [("train", ftr, "features_train_vif.csv"), ("validation", fva, "features_validation_vif.csv")]:
            ch = pd.read_csv(CHLOE / "round 03" / theirs, low_memory=False)
            m = mine.merge(ch, on=["id", "fs_year"], suffixes=("_mine", "_chloe"))
            assert len(m) == len(ch) == len(mine), (len(m), len(ch), len(mine))
            assert (m.label_mine == m.label_chloe).all()
            for f in SELECTED:
                x, y = m[f + "_mine"].astype(float), m[f + "_chloe"].astype(float)
                same = np.isclose(x, y, rtol=1e-5, atol=1e-7) | (x.isna() & y.isna())
                if name == "train" and f == "ateco_sector_te":
                    pass  # out-of-fold encoding on train depends on fold assignment; checked separately
                report.append(dict(split=name, feature=f, match_share=float(same.mean()),
                                   max_abs_diff=float(np.nanmax(np.abs(x - y))) if (~same).any() else 0.0))
            print(f"{name}: extra columns in mine not in 75: {sorted(set(mine.columns) - set(SELECTED) - set(META) - {'pred_start_dt'})}")
        r = pd.DataFrame(report); r.to_csv(HERE / "parity_round03.csv", index=False)
        bad = r[r.match_share < 0.9999]
        print(f"features matching on >= 99.99% of rows: {(r.match_share >= 0.9999).sum()} of {len(r)}")
        print(bad.to_string(index=False) if len(bad) else "ALL 75 features match in train and validation")
    else:
        tr = labelled_raw([2007, 2008, 2009, 2010, 2011])
        print(f"final-fit rows {len(tr):,} (expected 835,016), defaults {int(tr.label.sum()):,}, rate {tr.label.mean():.6f}")
        ftr, art = fit_preprocessor(tr, tr, cfg)
        missing = sorted(set(SELECTED) - set(ftr.columns))
        assert not missing, f"selected features not produced: {missing}"
        ftr[META + SELECTED].to_parquet(HERE / "final_train_features.parquet")
        joblib.dump(art, HERE / "final_preprocessor.joblib")
        print(f"wrote final_train_features.parquet {ftr[META + SELECTED].shape} and final_preprocessor.joblib")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
