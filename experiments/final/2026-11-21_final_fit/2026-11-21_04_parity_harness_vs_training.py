"""Parity test: PDs written by the harness (clean venv, pinned versions) vs PDs recomputed with the training
pipeline (vendored preprocess.transform_with_preprocessor + final model + the same calibration formula, in
the development env). Run on the full raw training file. Expected: identical up to float rounding.

Usage: python 2026-11-21_04_parity_harness_vs_training.py <harness_output_csv> [<variant name>]
(variant: compare against variants/<name>/lightgbm.txt + params.json instead of the main final model)
"""
import json
import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import expit, logit

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "vendor_7949207" / "src"))
from finance_project.data_processing.preprocess import transform_with_preprocessor  # noqa: E402

REPO = HERE.parents[2]
raw = pd.read_csv(REPO / "data/raw/train-2025.csv", low_memory=False)
assert not raw.duplicated(["id", "fs_year"]).any()
harness = pd.read_csv(sys.argv[1], header=None)[0].to_numpy()
assert len(harness) == len(raw)

art = joblib.load(HERE / "final_preprocessor.joblib")
VAR = sys.argv[2] if len(sys.argv) > 2 else None
MODEL = HERE / "variants" / VAR / "lightgbm.txt" if VAR else HERE / "final_lightgbm.txt"
PARAMS = HERE / "variants" / VAR / "params.json" if VAR else HERE / "final_params.json"
params = json.loads(PARAMS.read_text())
c = params["calibration"]
feats = transform_with_preprocessor(raw.drop(columns=["def_date"]), raw.drop(columns=["def_date"]), art)
booster = lgb.Booster(model_file=str(MODEL))
p = booster.predict(feats[params["feature_columns"]].astype(float), num_iteration=params["best_iteration"])
pd_train = np.clip(expit(c["anchor_a"] + c["slope_b"] * logit(np.clip(p, c["epsilon"], 1 - c["epsilon"])) + c["delta"]),
                   c["floor"], c["cap"])
ref = feats[["id", "fs_year"]].assign(pd_training_pipeline=pd_train)
cmp = raw[["id", "fs_year"]].assign(pd_harness=harness).merge(ref, on=["id", "fs_year"], how="left")
d = np.abs(cmp.pd_harness - cmp.pd_training_pipeline)
out = dict(rows=len(cmp), rows_scored_by_training_pipeline=int(cmp.pd_training_pipeline.notna().sum()),
           max_abs_diff=float(d.max()), rows_diff_gt_1e12=int((d > 1e-12).sum()), rows_diff_gt_1e9=int((d > 1e-9).sum()),
           mean_pd_harness=float(cmp.pd_harness.mean()))
(HERE / (f"variants/{VAR}/parity_harness_vs_training.json" if VAR else "parity_harness_vs_training.json")).write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=2))
