"""Developer verification of exported inference against saved round-03 outputs."""

from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/harness_submission"))
sys.path.insert(0, str(ROOT / "src"))
from inference import load_parameters, transform, calibrated_predictions, predict_frame, predict_csv
# Names required to load a CLI-created preprocessing artifact under __main__.
from finance_project.data_processing.preprocess import (
    PreprocessingArtifacts, PreprocessingConfig, transform_with_preprocessor,
)
import joblib


def main():
    parameters, booster = load_parameters()
    round_dir = ROOT / "data/processed/walk_forward_temporal/round_03"
    # Validate exported trees and calibration separately from historical feature CSVs.
    saved_features = pd.read_csv(round_dir / "features_test.csv", low_memory=False)
    expected = pd.read_csv(ROOT / "model_artifacts/temporal_firm_year_v2/round_03/lightgbm/test_predictions.csv")
    np.testing.assert_array_equal(saved_features[["id", "fs_year"]], expected[["id", "fs_year"]])
    pd_values = calibrated_predictions(saved_features[parameters["feature_columns"]], parameters, booster)
    error = np.max(np.abs(expected["pd_calibrated"] - pd_values))
    assert error < 1e-10, error
    print(f"PASS: frozen model/calibrator reproduce {len(expected):,} saved-feature predictions; max error {error:.3g}.")

    raw = pd.read_csv(round_dir / "history.csv", nrows=3000, low_memory=False).drop(columns="label", errors="ignore")
    transformed = transform(raw, parameters)
    original_artifacts = joblib.load(ROOT / "models/preprocessing/temporal_firm_year_round_03.joblib")
    original = transform_with_preprocessor(raw, raw, original_artifacts)
    np.testing.assert_array_equal(transformed.index, original.index)
    np.testing.assert_allclose(transformed, original[parameters["feature_columns"]], rtol=1e-12, atol=1e-8)
    print(f"PASS: {len(transformed):,} unlabeled rows match the original frozen preprocessing pipeline.")

    sample = raw.head(10).sample(frac=1, random_state=7).reset_index(drop=True)
    # Duplicate firm-year rows remain individually ordered predictions.
    sample = pd.concat([sample, sample.iloc[[0]]], ignore_index=True)
    expected_features = transform(sample, parameters)
    expected_pd = calibrated_predictions(expected_features, parameters, booster)
    with tempfile.TemporaryDirectory() as task_tmp:
        input_csv = Path(task_tmp) / "unlabeled.csv"
        output_csv = Path(task_tmp) / "output.csv"
        sample.to_csv(input_csv, index=False)
        predict_csv(input_csv, output_csv)
        result = pd.read_csv(output_csv, header=None)
        np.testing.assert_allclose(result[0], expected_pd, rtol=1e-12, atol=1e-12)
        assert len(result) == len(sample)
        # A nonpositive asset amount makes log(size) invalid and requires fallback.
        sample.loc[2, "asst_tot"] = -1
        invalid = transform(sample, parameters)
        assert 2 not in invalid.index and len(invalid) == len(sample) - 1
        sample.to_csv(input_csv, index=False)
        predict_csv(input_csv, output_csv)
        result = pd.read_csv(output_csv, header=None)[0].to_numpy()
        assert len(result) == len(sample)
        np.testing.assert_allclose(result[2], parameters["fallback_pd"], rtol=0, atol=1e-16)
        valid_pd = calibrated_predictions(invalid, parameters, booster)
        np.testing.assert_allclose(result[invalid.index], valid_pd, rtol=1e-12, atol=1e-12)
        # Input non-finites and multiple missing raw fields are imputed.
        sample.loc[3, ["AR", "roe", "cash_and_equiv"]] = [np.inf, np.nan, np.nan]
        assert np.isfinite(transform(sample, parameters).to_numpy()).all()
        # Invalid rows at both ends retain their positions, including nondefault indices.
        sample.loc[[0, len(sample) - 1], "asst_tot"] = 0
        sample.index = range(100, 100 + len(sample))
        result = predict_frame(sample, parameters, booster)
        assert len(result) == len(sample)
        assert (result[[0, 2, len(sample) - 1]] == parameters["fallback_pd"]).all()
        all_invalid = sample.copy()
        all_invalid["asst_tot"] = -1
        # No model call is needed when all rows require fallback.
        all_invalid_pd = predict_frame(all_invalid, parameters, booster=None)
        assert (all_invalid_pd == parameters["fallback_pd"]).all()
        sample.iloc[:0].to_csv(input_csv, index=False)
        predict_csv(input_csv, output_csv)
        assert output_csv.stat().st_size == 0
    print("PASS: unlabeled CSV, input order, duplicates, mixed/all-fallback rows, missing/inf handling, empty input.")


if __name__ == "__main__":
    main()
