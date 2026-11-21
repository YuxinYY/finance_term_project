"""Refit the selected LightGBM specification as one deployable model bundle.

This command is deliberately separate from walk-forward evaluation.  It fits
the preprocessing artifact and fixed-iteration LightGBM model on every row in
the labeled input, while Platt calibration is fitted only on saved out-of-time
raw predictions from a completed temporal LightGBM run.  Thus calibration is
not estimated from in-sample final-model predictions.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from finance_project.common import project_path, write_json
from finance_project.data_processing.preprocess import PreprocessingConfig, fit_preprocessor
from finance_project.model_development.lightgbm_model import (
    LightGBMConfig,
    fit_final_monotone_lightgbm,
)
from finance_project.model_development.run_walk_forward_models import (
    ProbabilityCalibrator,
    probability_metrics,
)


METADATA_COLUMNS = {"id", "fs_year", "pred_start_dt", "label"}


def load_out_of_time_calibration_predictions(calibration_run: str | Path) -> pd.DataFrame:
    """Load pooled raw test predictions from completed temporal LightGBM rounds.

    Each test observation was scored by a model that did not train on that
    fiscal year, so these predictions are suitable for fitting the final Platt
    mapping after model-family selection is complete.
    """
    run_dir = project_path(calibration_run)
    paths = sorted(run_dir.glob("round_*/lightgbm/test_predictions.csv"))
    if not paths:
        raise FileNotFoundError(
            "No LightGBM test-prediction files were found under "
            f"{run_dir}. Expected round_XX/lightgbm/test_predictions.csv."
        )

    frames: list[pd.DataFrame] = []
    for path in paths:
        frame = pd.read_csv(path, low_memory=False)
        required = {"label", "pd_raw"}
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"Calibration file {path} is missing columns: {missing}")
        frame = frame.copy()
        frame["source_round"] = path.parents[1].name
        frames.append(frame)

    pooled = pd.concat(frames, ignore_index=True)
    pooled["label"] = pd.to_numeric(pooled["label"], errors="raise").astype(int)
    pooled["pd_raw"] = pd.to_numeric(pooled["pd_raw"], errors="raise")
    pooled = pooled.replace([np.inf, -np.inf], np.nan).dropna(subset=["label", "pd_raw"])
    if set(pooled["label"].unique()) != {0, 1}:
        raise ValueError("Pooled calibration predictions must contain both binary label classes.")
    return pooled


def _read_labeled_input(path: str | Path, target_column: str) -> pd.DataFrame:
    source = project_path(path)
    if not source.exists():
        raise FileNotFoundError(f"Labeled input data not found: {source}")
    frame = pd.read_csv(source, low_memory=False)
    if target_column not in frame.columns:
        raise ValueError(f"Labeled input is missing target column '{target_column}'.")
    if frame[target_column].isna().any():
        raise ValueError("Production refit requires a complete, mature target label for every input row.")
    return frame


def fit_production_lightgbm(
    labeled_data: pd.DataFrame,
    calibration_predictions: pd.DataFrame,
    *,
    preprocessing_config: PreprocessingConfig,
    lightgbm_config: LightGBMConfig,
) -> tuple[object, object, ProbabilityCalibrator, dict[str, object]]:
    """Fit preprocessing, final LightGBM, and an out-of-time Platt calibrator."""
    target_column = preprocessing_config.target_column
    history = labeled_data.drop(columns=target_column, errors="ignore")
    transformed, preprocessor = fit_preprocessor(labeled_data, history, preprocessing_config)
    feature_columns = [column for column in transformed.columns if column not in METADATA_COLUMNS]
    features = transformed.reindex(columns=feature_columns).apply(pd.to_numeric, errors="coerce")
    target = pd.to_numeric(transformed[target_column], errors="raise").astype(int)
    if set(target.unique()) != {0, 1}:
        raise ValueError("Production LightGBM training requires both binary label classes.")

    model, model_details = fit_final_monotone_lightgbm(features, target, lightgbm_config)
    calibrator = ProbabilityCalibrator.fit(
        calibration_predictions["pd_raw"].to_numpy(),
        calibration_predictions["label"],
        method="platt",
    )
    calibration_pd = calibrator.transform(calibration_predictions["pd_raw"].to_numpy())
    summary: dict[str, object] = {
        "training_rows": int(len(transformed)),
        "training_default_rate": float(target.mean()),
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "model_details": model_details,
        "calibration_rows": int(len(calibration_predictions)),
        "calibration_source_rounds": sorted(calibration_predictions["source_round"].unique().tolist()),
        "calibration_raw_metrics": probability_metrics(
            calibration_predictions["label"], calibration_predictions["pd_raw"].to_numpy()
        ),
        "calibration_platt_metrics": probability_metrics(
            calibration_predictions["label"], calibration_pd
        ),
    }
    return preprocessor, model, calibrator, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/processed/labeled_data.csv")
    parser.add_argument(
        "--calibration-run",
        default="model_artifacts/temporal_firm_year_v1",
        help="Completed temporal LightGBM run containing out-of-time test_predictions.csv files.",
    )
    parser.add_argument("--output-dir", default="model_artifacts/production_lightgbm_v1")
    parser.add_argument("--id-column", default="id")
    parser.add_argument("--year-column", default="fs_year")
    parser.add_argument("--target-column", default="label")
    parser.add_argument("--sector-column", default="ateco_sector")
    parser.add_argument("--legal-column", default="legal_struct")
    parser.add_argument(
        "--feature-mode",
        choices=["firm_year", "decayed_history"],
        default="firm_year",
        help="Use firm_year to match the selected temporal_firm_year LightGBM experiment.",
    )
    parser.add_argument("--target-encoding-smoothing", type=float, default=20.0)
    parser.add_argument("--target-encoding-folds", type=int, default=5)
    parser.add_argument("--n-estimators", type=int, default=314)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--num-leaves", type=int, default=15)
    parser.add_argument("--min-child-samples", type=int, default=200)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    parser.add_argument("--reg-lambda", type=float, default=1.0)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preprocessing_config = PreprocessingConfig(
        id_column=args.id_column,
        year_column=args.year_column,
        target_column=args.target_column,
        sector_column=args.sector_column,
        legal_column=args.legal_column,
        feature_mode=args.feature_mode,
        target_encoding_smoothing=args.target_encoding_smoothing,
        target_encoding_folds=args.target_encoding_folds,
        random_state=args.random_state,
    )
    lightgbm_config = LightGBMConfig(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        min_child_samples=args.min_child_samples,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_lambda=args.reg_lambda,
        random_state=args.random_state,
    )
    labeled_data = _read_labeled_input(args.input, args.target_column)
    calibration_predictions = load_out_of_time_calibration_predictions(args.calibration_run)
    preprocessor, model, calibrator, summary = fit_production_lightgbm(
        labeled_data,
        calibration_predictions,
        preprocessing_config=preprocessing_config,
        lightgbm_config=lightgbm_config,
    )

    output_dir = project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    preprocessor_path = output_dir / "preprocessor.joblib"
    model_path = output_dir / "lightgbm_model.joblib"
    calibrator_path = output_dir / "platt_calibrator.joblib"
    joblib.dump(preprocessor, preprocessor_path)
    joblib.dump(model, model_path)
    joblib.dump(calibrator, calibrator_path)
    manifest_path = write_json(
        output_dir / "training_manifest.json",
        {
            "model_family": "lightgbm",
            "purpose": "production_refit",
            "input": str(project_path(args.input)),
            "fiscal_year_range": [
                int(pd.to_numeric(labeled_data[args.year_column], errors="raise").min()),
                int(pd.to_numeric(labeled_data[args.year_column], errors="raise").max()),
            ],
            "preprocessing_config": asdict(preprocessing_config),
            "lightgbm_config": asdict(lightgbm_config),
            "calibration_run": str(project_path(args.calibration_run)),
            "calibration_method": "platt",
            "calibration_note": (
                "Platt calibration is fitted on pooled out-of-time raw LightGBM predictions "
                "from the completed temporal run, never on final-model in-sample predictions."
            ),
            "artifacts": {
                "preprocessor": str(preprocessor_path),
                "lightgbm_model": str(model_path),
                "platt_calibrator": str(calibrator_path),
            },
            **summary,
        },
    )
    print(f"Wrote production preprocessor: {preprocessor_path}")
    print(f"Wrote production LightGBM model: {model_path}")
    print(f"Wrote production Platt calibrator: {calibrator_path}")
    print(f"Wrote production manifest: {manifest_path}")


if __name__ == "__main__":
    main()
