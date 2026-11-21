"""Developer checks for source synchronization, notebook parity and row independence."""

import ast
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
sys.path.insert(0, str(PROJECT / "src"))

import anomaly_rules
import feature_rules
import missing_rules
from finance_project.data_processing import preprocess


def main():
    for notebook, module in (("01_flag_anomalies", anomaly_rules),
                             ("02_missing_values", missing_rules)):
        content = json.loads((PROJECT / f"notebooks/features/{notebook}.ipynb").read_text())
        source = "\n\n".join("".join(c["source"]) for c in content["cells"]
                               if "export" in c.get("metadata", {}).get("tags", []))
        generated = ast.parse(Path(module.__file__).read_text())
        # Ignore only the generated header and dependency imports.
        actual = ast.Module(body=generated.body[4:], type_ignores=[])
        assert ast.dump(actual) == ast.dump(ast.parse(source)), notebook

    base = pd.read_csv(PROJECT / "data/raw/train_demo_subset.csv", nrows=1).iloc[0].to_dict()
    records = []
    for group_number, sector in enumerate((10, 41, 45, 49, 68), start=1):
        for _ in range(group_number):
            row = dict(base, id=len(records) + 1, fs_year=2010,
                       ateco_sector=sector, profit=20, eqty_tot=10, roe=999, liab_lt=5)
            records.append(row)
    train = pd.DataFrame(records)
    medians = missing_rules.sector_medians(train)
    assert set(medians.index) == {"H", "L", "ALL"}, medians.index
    pd.testing.assert_frame_equal(medians, preprocess._missing["sector_medians"](train))
    sample = train.iloc[:4].copy().reset_index(drop=True)
    sample.loc[0, ["profit", "eqty_tot"]] = [-10, -5]
    sample.loc[1, ["profit", "eqty_tot"]] = [10, 0]
    sample.loc[2, "ateco_sector"] = 999
    sample.loc[2, "AR"] = np.nan
    sample.loc[3, "liab_lt"] = np.nan
    filled = missing_rules.fill_missing(anomaly_rules.flag_anomalies(sample), medians)
    assert filled.loc[0, "roe"] == 200 and filled.loc[0, "roe_negative"] == 1
    assert filled.loc[1, "roe"] == 1000 and filled.loc[1, "roe_negative"] == 0
    assert filled.loc[3, "liab_lt"] == 5
    config = preprocess.PreprocessingConfig(feature_mode="firm_year")
    original = preprocess.clean_and_impute(sample, medians, config)
    pd.testing.assert_frame_equal(filled, original.drop(columns="ateco_industry_group"))
    pd.testing.assert_frame_equal(feature_rules.make_period_features(filled, config),
                                  preprocess.make_period_features(filled, config))
    for index in sample.index:
        alone = missing_rules.fill_missing(anomaly_rules.flag_anomalies(sample.loc[[index]]), medians)
        pd.testing.assert_frame_equal(alone, filled.loc[[index]])
    # An earlier zero from the same firm must no longer fill the later row.
    pair = sample.iloc[[3, 3]].copy().reset_index(drop=True)
    pair["id"] = 9999
    pair["fs_year"] = [2010, 2011]
    pair["liab_lt"] = [0, np.nan]
    assert missing_rules.fill_missing(pair, medians).loc[1, "liab_lt"] == 5
    print("PASS: exact notebook exports; smallest-three fallback; ROE and flag; no cross-row imputation; preprocessing parity.")

    result = subprocess.run(
        [sys.executable, str(ROOT / "build_submission.py"),
         "--preprocessor", str(PROJECT / "models/preprocessing/temporal_firm_year_round_03.joblib"),
         "--model", str(PROJECT / "model_artifacts/temporal_firm_year_v2/round_03/lightgbm/model.joblib")],
        capture_output=True, text=True,
    )
    assert result.returncode != 0 and "stale" in result.stderr, result.stderr
    assert not (ROOT / "artifacts").exists()
    print("PASS: incompatible old artifacts are rejected before any deployment files are written.")


if __name__ == "__main__":
    main()
