"""Run WoE/logistic and monotone-LightGBM models across walk-forward rounds.

Validation rows are used for model selection or early stopping and for fitting a
Platt probability calibrator.  Test rows are never used until final scoring.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from finance_project.common import project_path, write_json
from finance_project.model_development.lightgbm_model import LightGBMConfig, fit_monotone_lightgbm
from finance_project.model_development.woe_logistic import WOELogisticConfig, fit_woe_logistic


METADATA_COLUMNS = ("id", "fs_year", "pred_start_dt", "label")
EPSILON = 1e-6


@dataclass(frozen=True)
class RunnerConfig:
    data_dir: str = "data/processed/walk_forward"
    model_dir: str = "models/model_development"
    report_dir: str = "reports/model_development"
    calibration: str = "platt"
    random_state: int = 42


@dataclass
class ProbabilityCalibrator:
    """A validation-fitted mapping from raw classifier probabilities to PDs."""

    method: str
    estimator: Any | None = None

    @staticmethod
    def _logit(probabilities: np.ndarray) -> np.ndarray:
        clipped = np.clip(np.asarray(probabilities, dtype=float), EPSILON, 1 - EPSILON)
        return np.log(clipped / (1 - clipped))

    @classmethod
    def fit(cls, probabilities: np.ndarray, target: pd.Series, method: str) -> "ProbabilityCalibrator":
        if method not in {"none", "platt", "isotonic"}:
            raise ValueError("calibration must be one of: none, platt, isotonic.")
        if method == "none":
            return cls(method="none")
        if pd.Series(target).nunique() != 2:
            return cls(method="none")
        if method == "platt":
            estimator = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1_000)
            estimator.fit(cls._logit(probabilities).reshape(-1, 1), target)
        else:
            estimator = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
            estimator.fit(probabilities, target)
        return cls(method=method, estimator=estimator)

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        probabilities = np.asarray(probabilities, dtype=float)
        if self.method == "none" or self.estimator is None:
            return np.clip(probabilities, EPSILON, 1 - EPSILON)
        if self.method == "platt":
            return self.estimator.predict_proba(self._logit(probabilities).reshape(-1, 1))[:, 1]
        return np.asarray(self.estimator.predict(probabilities), dtype=float)


def _safe_metric(function: Any, actual: pd.Series, probabilities: np.ndarray) -> float | None:
    if pd.Series(actual).nunique() != 2:
        return None
    return float(function(actual, probabilities))


def _calibration_slope(actual: pd.Series, probabilities: np.ndarray) -> float | None:
    if pd.Series(actual).nunique() != 2:
        return None
    estimator = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1_000)
    estimator.fit(ProbabilityCalibrator._logit(probabilities).reshape(-1, 1), actual)
    return float(estimator.coef_[0, 0])


def probability_metrics(actual: pd.Series, probabilities: np.ndarray) -> dict[str, float | None]:
    """Return discrimination and PD-quality metrics for a binary target."""
    probabilities = np.clip(np.asarray(probabilities, dtype=float), EPSILON, 1 - EPSILON)
    actual = pd.Series(actual).astype(int)
    return {
        "rows": int(len(actual)),
        "defaults": int(actual.sum()),
        "default_rate": float(actual.mean()),
        "mean_predicted_pd": float(probabilities.mean()),
        "calibration_in_the_large": float(probabilities.mean() - actual.mean()),
        "calibration_slope": _calibration_slope(actual, probabilities),
        "roc_auc": _safe_metric(roc_auc_score, actual, probabilities),
        "pr_auc": _safe_metric(average_precision_score, actual, probabilities),
        "log_loss": float(log_loss(actual, probabilities, labels=[0, 1])),
        "brier_score": float(brier_score_loss(actual, probabilities)),
    }


def pd_deciles(actual: pd.Series, probabilities: np.ndarray) -> list[dict[str, float | int]]:
    """Observed-versus-predicted PD table, ordered from lowest to highest score."""
    table = pd.DataFrame({"label": pd.Series(actual).to_numpy(), "pd": probabilities})
    ranks = table["pd"].rank(method="first")
    bins = min(10, len(table))
    table["decile"] = pd.qcut(ranks, q=bins, labels=False, duplicates="drop") + 1
    summary = (
        table.groupby("decile", observed=True)
        .agg(rows=("label", "size"), defaults=("label", "sum"), observed_default_rate=("label", "mean"), mean_pd=("pd", "mean"))
        .reset_index()
    )
    return [
        {
            "decile": int(row.decile),
            "rows": int(row.rows),
            "defaults": int(row.defaults),
            "observed_default_rate": float(row.observed_default_rate),
            "mean_pd": float(row.mean_pd),
        }
        for row in summary.itertuples(index=False)
    ]


def load_feature_splits(round_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """Load one feature round and align validation/test columns to training columns."""
    round_dir = Path(round_dir)
    splits = {
        split: pd.read_csv(round_dir / f"features_{split}.csv", low_memory=False)
        for split in ("train", "validation", "test")
    }
    for name, frame in splits.items():
        if "label" not in frame:
            raise ValueError(f"{name} split lacks the required 'label' column.")

    feature_columns = [column for column in splits["train"].columns if column not in METADATA_COLUMNS]
    if not feature_columns:
        raise ValueError(f"No model features found in {round_dir}.")

    for split_name in ("validation", "test"):
        frame = splits[split_name]
        extras = sorted(set(frame.columns) - set(feature_columns) - set(METADATA_COLUMNS))
        if extras:
            raise ValueError(f"{split_name} has unexpected model columns: {extras}")
        for column in feature_columns:
            if column not in frame:
                # Legal-form dummies being absent means that category did not
                # occur in the split; their correct value is zero.
                frame[column] = 0 if column.startswith("legal_") else np.nan
        splits[split_name] = frame
    return splits["train"], splits["validation"], splits["test"], feature_columns


def _features_and_target(frame: pd.DataFrame, feature_columns: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    features = frame.reindex(columns=feature_columns).apply(pd.to_numeric, errors="coerce")
    target = pd.to_numeric(frame["label"], errors="raise").astype(int)
    if set(target.unique()) - {0, 1}:
        raise ValueError("label must be binary 0/1.")
    return features, target


def _prediction_frame(frame: pd.DataFrame, raw_pd: np.ndarray, calibrated_pd: np.ndarray) -> pd.DataFrame:
    columns = [column for column in METADATA_COLUMNS if column in frame.columns]
    output = frame[columns].copy()
    output["pd_raw"] = raw_pd
    output["pd_calibrated"] = calibrated_pd
    return output


def run_model_for_round(
    *,
    round_dir: str | Path,
    model_family: str,
    config: RunnerConfig | None = None,
    woe_config: WOELogisticConfig | None = None,
    lightgbm_config: LightGBMConfig | None = None,
) -> dict[str, Any]:
    """Train one model family for one round and write its artifacts and reports."""
    config = config or RunnerConfig()
    if model_family not in {"woe_logistic", "lightgbm"}:
        raise ValueError("model_family must be 'woe_logistic' or 'lightgbm'.")

    train, validation, test, feature_columns = load_feature_splits(round_dir)
    x_train, y_train = _features_and_target(train, feature_columns)
    x_validation, y_validation = _features_and_target(validation, feature_columns)
    x_test, y_test = _features_and_target(test, feature_columns)

    if model_family == "woe_logistic":
        model, model_details = fit_woe_logistic(x_train, y_train, x_validation, y_validation, woe_config)
    else:
        model, model_details = fit_monotone_lightgbm(
            x_train, y_train, x_validation, y_validation, lightgbm_config
        )

    validation_raw = model.predict_proba(x_validation)
    calibrator = ProbabilityCalibrator.fit(validation_raw, y_validation, config.calibration)
    validation_calibrated = calibrator.transform(validation_raw)
    test_raw = model.predict_proba(x_test)
    test_calibrated = calibrator.transform(test_raw)

    round_name = Path(round_dir).name
    model_path = project_path(config.model_dir) / round_name / f"{model_family}.joblib"
    report_path = project_path(config.report_dir) / round_name / f"{model_family}.json"
    prediction_path = project_path(config.report_dir) / round_name / f"{model_family}_test_predictions.csv"
    validation_prediction_path = project_path(config.report_dir) / round_name / f"{model_family}_validation_predictions.csv"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model_family": model_family,
            "model": model,
            "calibrator": calibrator,
            "feature_columns": feature_columns,
            "runner_config": asdict(config),
            "model_details": model_details,
        },
        model_path,
    )
    _prediction_frame(test, test_raw, test_calibrated).to_csv(prediction_path, index=False)
    _prediction_frame(validation, validation_raw, validation_calibrated).to_csv(
        validation_prediction_path, index=False
    )

    payload: dict[str, Any] = {
        "round": round_name,
        "model_family": model_family,
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "model_details": model_details,
        "calibration": {
            "method": calibrator.method,
            "note": "The validation partition is used for fitting this calibrator after model selection/early stopping.",
        },
        "validation": {
            "raw": probability_metrics(y_validation, validation_raw),
            "calibrated": probability_metrics(y_validation, validation_calibrated),
        },
        "test": {
            "raw": probability_metrics(y_test, test_raw),
            "calibrated": probability_metrics(y_test, test_calibrated),
            "calibrated_pd_deciles": pd_deciles(y_test, test_calibrated),
        },
        "artifacts": {
            "model": str(model_path),
            "test_predictions": str(prediction_path),
            "validation_predictions": str(validation_prediction_path),
        },
    }
    write_json(report_path, payload)
    return payload


def _parse_rounds(value: str) -> list[int]:
    rounds = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not rounds or any(round_number < 1 for round_number in rounds):
        raise argparse.ArgumentTypeError("rounds must be comma-separated positive integers, e.g. 1,2,3,4")
    return rounds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/processed/walk_forward")
    parser.add_argument("--models", choices=["woe_logistic", "lightgbm", "both"], default="both")
    parser.add_argument("--rounds", type=_parse_rounds, default=[1, 2, 3, 4])
    parser.add_argument("--model-dir", default="models/model_development")
    parser.add_argument("--report-dir", default="reports/model_development")
    parser.add_argument("--calibration", choices=["none", "platt", "isotonic"], default="platt")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--woe-bins", type=int, default=10)
    parser.add_argument("--woe-smoothing", type=float, default=0.5)
    parser.add_argument("--lgbm-estimators", type=int, default=5_000)
    parser.add_argument("--lgbm-learning-rate", type=float, default=0.03)
    parser.add_argument("--lgbm-num-leaves", type=int, default=15)
    parser.add_argument("--lgbm-min-child-samples", type=int, default=200)
    parser.add_argument("--early-stopping-rounds", type=int, default=150)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runner_config = RunnerConfig(
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        report_dir=args.report_dir,
        calibration=args.calibration,
        random_state=args.random_state,
    )
    woe_config = WOELogisticConfig(n_bins=args.woe_bins, smoothing=args.woe_smoothing, random_state=args.random_state)
    lightgbm_config = LightGBMConfig(
        n_estimators=args.lgbm_estimators,
        learning_rate=args.lgbm_learning_rate,
        num_leaves=args.lgbm_num_leaves,
        min_child_samples=args.lgbm_min_child_samples,
        early_stopping_rounds=args.early_stopping_rounds,
        random_state=args.random_state,
    )
    model_families = ["woe_logistic", "lightgbm"] if args.models == "both" else [args.models]
    summaries = []
    for round_number in args.rounds:
        round_dir = project_path(args.data_dir) / f"round_{round_number:02d}"
        for model_family in model_families:
            summary = run_model_for_round(
                round_dir=round_dir,
                model_family=model_family,
                config=runner_config,
                woe_config=woe_config,
                lightgbm_config=lightgbm_config,
            )
            summaries.append(
                {
                    "round": summary["round"],
                    "model_family": model_family,
                    "test_roc_auc": summary["test"]["calibrated"]["roc_auc"],
                    "test_pr_auc": summary["test"]["calibrated"]["pr_auc"],
                    "test_mean_pd": summary["test"]["calibrated"]["mean_predicted_pd"],
                }
            )
            print(f"Completed {model_family} for {round_dir.name}")
    write_json(project_path(args.report_dir) / "summary.json", {"runs": summaries})


if __name__ == "__main__":
    main()
