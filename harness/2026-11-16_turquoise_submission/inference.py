"""Inference only: local frozen parameters, preprocessing, and calibrated PDs."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import lightgbm as lgb
import numpy as np
import pandas as pd

import anomaly_rules
import feature_rules
import missing_rules


ROOT = Path(__file__).resolve().parent


def load_parameters() -> tuple[dict, lgb.Booster]:
    manifest = json.loads((ROOT / "artifacts/manifest.json").read_text())
    for relative, expected in manifest["bundle_sha256"].items():
        actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        if actual != expected:
            # Warn only: a line-ending conversion on unzip must not stop the run.
            print(f"WARNING: bundle file differs from manifest checksum: {relative}", file=sys.stderr)
    parameters = json.loads((ROOT / "artifacts/preprocessing.json").read_text())
    fallback_pd = parameters["fallback_pd"]
    if not np.isfinite(fallback_pd) or not 0 <= fallback_pd <= 1:
        raise ValueError("The frozen fallback PD must be finite and between zero and one.")
    if parameters["config"]["feature_mode"] != "firm_year":
        raise ValueError("This harness requires the frozen firm_year preprocessor.")
    booster = lgb.Booster(model_file=str(ROOT / "artifacts/lightgbm.txt"))
    # LightGBM normalizes whitespace in names when it stores the tree model.
    if booster.feature_name() != parameters["model_feature_names"]:
        raise ValueError("Model and preprocessing feature orders do not match.")
    return parameters, booster


def transform(frame: pd.DataFrame, parameters: dict) -> pd.DataFrame:
    """Return valid model rows, indexed by their original input positions.

    Rows excluded here receive the frozen fallback PD in predict_frame.
    """
    frame = frame.copy().reset_index(drop=True)
    frame.columns = [str(c).lstrip("\ufeff").strip() for c in frame.columns]
    if frame.columns.duplicated().any():
        raise ValueError("Input contains duplicate column names after normalization.")
    required = set(feature_rules.FEATURE_SCHEMA["raw_required"]) | set(
        feature_rules.FEATURE_SCHEMA["impute_columns"]
    )
    missing = sorted(required.difference(frame.columns))
    if missing:
        # Never halt: an absent column is treated as all-missing and imputed like any NaN.
        print(f"WARNING: input lacks columns {missing}; treating them as missing values.", file=sys.stderr)
    # Dates, targets, index exports, and unrelated columns do not enter features.
    raw_columns = sorted(required | {"roa"})
    frame = frame.reindex(columns=raw_columns)
    config = SimpleNamespace(**parameters["config"])
    numeric_columns = [c for c in raw_columns if c != config.legal_column]
    frame[numeric_columns] = frame[numeric_columns].apply(pd.to_numeric, errors="coerce")
    raw_missing = int(frame[numeric_columns].isna().sum().sum())
    raw_infinite = int(np.isinf(frame[numeric_columns].to_numpy(dtype=float)).sum())
    print(f"Raw check: {raw_missing:,} missing, {raw_infinite:,} infinite numeric values.", file=sys.stderr)
    # Non-finite reported inputs are unavailable values for imputation.
    frame[numeric_columns] = frame[numeric_columns].replace([np.inf, -np.inf], np.nan)
    flagged = anomaly_rules.flag_anomalies(frame)
    print(f"Anomaly check: {int(flagged['neg_bs_flag'].sum()):,} rows flagged.", file=sys.stderr)
    medians = pd.DataFrame(**parameters["medians"])
    filled = missing_rules.fill_missing(flagged, medians)
    filled[config.legal_column] = filled[config.legal_column].fillna("Unknown").astype(str)
    features = feature_rules.make_period_features(filled, config)
    lookup = missing_rules._ateco_industry_lookup()
    industry = missing_rules._ateco_industry_group(features[config.sector_column], lookup)
    features["ateco_sector_te"] = industry.map(parameters["industry_mapping"]).fillna(
        parameters["industry_global_mean"]
    )
    legal = features[config.legal_column]
    for category in parameters["legal_categories"]:
        features[f"legal_{category}"] = legal.eq(category).astype(int)
    region = feature_rules._region(features["HQ_city"])
    for category in parameters["city_categories"]:
        features[f"city_{category}"] = region.eq(category).astype(int)
    # Restore CSV order after the firm/year sort used by the notebook rules.
    features = features.reindex(frame.index)[parameters["feature_columns"]].apply(
        pd.to_numeric, errors="coerce"
    )
    finite = np.isfinite(features.to_numpy(dtype=float)).all(axis=1)
    print(
        f"Feature check: {int(features.isna().sum().sum()):,} missing, "
        f"{int(np.isinf(features.to_numpy(dtype=float)).sum()):,} infinite values; "
        f"{int((~finite).sum()):,} of {len(frame):,} rows require fallback PDs.",
        file=sys.stderr,
    )
    return features.loc[finite].copy()


def calibrated_predictions(features: pd.DataFrame, parameters: dict, booster: lgb.Booster) -> np.ndarray:
    if features.empty:
        return np.empty(0, dtype=float)
    raw = booster.predict(features, num_iteration=parameters["best_iteration"], num_threads=1)
    if not np.isfinite(raw).all():
        raise ValueError("Frozen model produced non-finite predictions.")
    calibration = parameters["calibration"]
    clipped = np.clip(raw, calibration["epsilon"], 1 - calibration["epsilon"])
    if calibration["method"] == "platt":
        logit = np.log(clipped / (1 - clipped))
        score = calibration["coefficient"] * logit + calibration["intercept"]
        # Stable sigmoid without overflow at large positive/negative scores.
        pd_values = np.exp(-np.logaddexp(0, -score))
    else:
        pd_values = clipped
    if not np.isfinite(pd_values).all() or ((pd_values < 0) | (pd_values > 1)).any():
        raise ValueError("Calibration produced invalid probabilities.")
    return pd_values


def predict_frame(frame: pd.DataFrame, parameters: dict, booster: lgb.Booster) -> np.ndarray:
    """Return one PD per input row, using the frozen prior for invalid features."""
    features = transform(frame, parameters)
    # The prior is already a PD; do not apply the model's Platt mapping to it.
    predictions = np.full(len(frame), parameters["fallback_pd"], dtype=float)
    predictions[features.index.to_numpy(dtype=int)] = calibrated_predictions(features, parameters, booster)
    print(
        f"Predictions: {len(features):,} model PDs, {len(frame) - len(features):,} fallback PDs "
        f"(frozen training default rate {parameters['fallback_pd']:.8%}).",
        file=sys.stderr,
    )
    return predictions


def predict_csv(input_csv: Path, output_csv: Path) -> None:
    parameters, booster = load_parameters()
    frame = pd.read_csv(input_csv, low_memory=False, encoding="utf-8-sig")
    try:
        predictions = predict_frame(frame, parameters, booster)
    except Exception as error:  # last-resort guard: the assignment requires a PD for every row
        print(f"ERROR in pipeline ({error!r}); writing the frozen fallback PD for all "
              f"{len(frame):,} rows.", file=sys.stderr)
        predictions = np.full(len(frame), parameters["fallback_pd"], dtype=float)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.Series(predictions).to_csv(output_csv, index=False, header=False, float_format="%.17g")
    print(f"Wrote {len(predictions):,} PDs to {output_csv}.", file=sys.stderr)
