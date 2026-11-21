"""Create SHAP and RiskCalc-style feature-impact outputs for production LightGBM.

The command selects up to two interpretable features in each Feature Schema
group using mean absolute TreeSHAP values on a representative stratified sample.
For every selected feature it then uses the full labeled population to create
quantile-bin tables and charts of observed 12-month default frequency and mean
SHAP contribution.  These are descriptive model-interpretation artifacts, not
new out-of-sample performance estimates.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from finance_project.common import project_path
from finance_project.data_processing.preprocess import (
    FEATURE_GROUPS,
    PreprocessingArtifacts,
    transform_with_preprocessor,
)


METADATA_COLUMNS = {"id", "fs_year", "pred_start_dt", "label"}
CATEGORICAL_FAMILIES = {"legal_*": "legal_structure", "city_*": "HQ_city"}


def _read_labeled_data(path: str | Path, target_column: str) -> pd.DataFrame:
    source = project_path(path)
    if not source.exists():
        raise FileNotFoundError(f"Labeled data not found: {source}")
    frame = pd.read_csv(source, low_memory=False)
    if target_column not in frame.columns or frame[target_column].isna().any():
        raise ValueError("Interpretation requires a complete labeled input dataset.")
    return frame


def _load_production_bundle(production_dir: str | Path) -> tuple[PreprocessingArtifacts, Any, Any]:
    directory = project_path(production_dir)
    paths = {
        "preprocessor": directory / "preprocessor.joblib",
        "model": directory / "lightgbm_model.joblib",
        "calibrator": directory / "platt_calibrator.joblib",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Production bundle is incomplete: {missing}")
    preprocessor = joblib.load(paths["preprocessor"])
    model = joblib.load(paths["model"])
    calibrator = joblib.load(paths["calibrator"])
    if not isinstance(preprocessor, PreprocessingArtifacts):
        raise TypeError("preprocessor.joblib is not a PreprocessingArtifacts object.")
    if not hasattr(model, "estimator") or not hasattr(model, "feature_columns"):
        raise TypeError("lightgbm_model.joblib is not a compatible LightGBMModel object.")
    return preprocessor, model, calibrator


def _stratified_sample_positions(
    frame: pd.DataFrame,
    *,
    year_column: str,
    target_column: str,
    sample_size: int,
    random_state: int,
) -> np.ndarray:
    """Sample proportionally by fiscal year and target, preserving prevalence."""
    if sample_size >= len(frame):
        return np.arange(len(frame))
    if sample_size < 2:
        raise ValueError("sample_size must be at least 2.")
    strata = frame[year_column].astype(str) + "__" + frame[target_column].astype(str)
    if strata.value_counts().min() < 2:
        return frame.sample(n=sample_size, random_state=random_state).index.to_numpy()
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=sample_size, random_state=random_state)
    _, sample_positions = next(splitter.split(np.zeros(len(frame)), strata))
    return sample_positions


def _tree_shap_values(estimator: Any, features: pd.DataFrame) -> np.ndarray:
    """Return a two-dimensional positive-class TreeSHAP matrix."""
    try:
        import shap
    except ImportError as error:  # pragma: no cover - dependency is declared in pyproject
        raise ImportError("SHAP is required. Install project dependencies before interpretation.") from error

    explainer = shap.TreeExplainer(estimator)
    values = explainer.shap_values(features, check_additivity=False)
    if isinstance(values, list):
        values = values[1] if len(values) > 1 else values[0]
    if hasattr(values, "values"):
        values = values.values
    values = np.asarray(values)
    if values.ndim == 3:
        # Recent SHAP versions may return (rows, features, classes).
        values = values[:, :, 1] if values.shape[-1] > 1 else values[:, :, 0]
    if values.ndim != 2:
        raise ValueError(f"Expected a 2D SHAP matrix, received shape {values.shape}.")
    return values


def _group_items(feature_columns: list[str]) -> list[dict[str, Any]]:
    """Map Feature Schema groups to exact features and one-hot feature families."""
    rows: list[dict[str, Any]] = []
    for group, requirements in FEATURE_GROUPS.items():
        for requirement in requirements:
            if requirement.endswith("*"):
                columns = [column for column in feature_columns if column.startswith(requirement[:-1])]
                if not columns:
                    continue
                rows.append(
                    {
                        "group": group,
                        "feature": CATEGORICAL_FAMILIES.get(requirement, requirement[:-1]),
                        "feature_type": "categorical_family",
                        "model_columns": columns,
                        # Continuous ratios are the target of RiskCalc-style curves.
                        "selection_eligible": False,
                    }
                )
            elif requirement in feature_columns:
                rows.append(
                    {
                        "group": group,
                        "feature": requirement,
                        "feature_type": "numeric_or_binary",
                        "model_columns": [requirement],
                        "selection_eligible": True,
                    }
                )
    return rows


def grouped_shap_importance(
    features: pd.DataFrame,
    shap_values: np.ndarray,
) -> pd.DataFrame:
    """Calculate individual/family SHAP importance within each schema group."""
    feature_columns = list(features.columns)
    if shap_values.shape != features.shape:
        raise ValueError("SHAP matrix shape does not match the feature matrix.")
    column_to_position = {column: position for position, column in enumerate(feature_columns)}
    rows: list[dict[str, Any]] = []
    for item in _group_items(feature_columns):
        positions = [column_to_position[column] for column in item["model_columns"]]
        contribution = shap_values[:, positions].sum(axis=1)
        if len(positions) == 1:
            series = features[item["model_columns"][0]]
            n_non_missing = int(series.notna().sum())
            n_unique = int(series.nunique(dropna=True))
        else:
            n_non_missing = int(features[item["model_columns"]].notna().any(axis=1).sum())
            n_unique = int(features[item["model_columns"]].drop_duplicates().shape[0])
        rows.append(
            {
                "group": item["group"],
                "feature": item["feature"],
                "feature_type": item["feature_type"],
                "n_model_columns": len(item["model_columns"]),
                "model_columns": "|".join(item["model_columns"]),
                "mean_abs_shap": float(np.abs(contribution).mean()),
                "mean_shap": float(contribution.mean()),
                "sample_non_missing": n_non_missing,
                "sample_unique_values": n_unique,
                "selection_eligible": bool(item["selection_eligible"] and n_unique > 1),
            }
        )
    output = pd.DataFrame(rows)
    output["rank_in_group"] = output.groupby("group")["mean_abs_shap"].rank(
        method="first", ascending=False
    ).astype(int)
    return output.sort_values(["group", "rank_in_group", "feature"]).reset_index(drop=True)


def select_features(importance: pd.DataFrame, top_features: int) -> pd.DataFrame:
    """Select the strongest one or two interpretable numeric features per group."""
    if top_features not in {1, 2}:
        raise ValueError("top_features must be either 1 or 2.")
    eligible = importance.loc[importance["selection_eligible"]].copy()
    eligible["selection_rank"] = eligible.groupby("group")["mean_abs_shap"].rank(
        method="first", ascending=False
    ).astype(int)
    selected = eligible.loc[eligible["selection_rank"] <= top_features].copy()
    selected["selected"] = True
    return selected.sort_values(["group", "selection_rank", "feature"]).reset_index(drop=True)


def _wilson_interval(defaults: pd.Series, counts: pd.Series) -> tuple[pd.Series, pd.Series]:
    z = 1.96
    rate = defaults.div(counts)
    denominator = 1 + z**2 / counts
    center = (rate + z**2 / (2 * counts)).div(denominator)
    margin = (
        z
        * np.sqrt((rate * (1 - rate) + z**2 / (4 * counts)).div(counts))
        / denominator
    )
    return (center - margin).clip(lower=0), (center + margin).clip(upper=1)


def _rolling_weighted_mean(values: pd.Series, weights: pd.Series, window: int = 5) -> pd.Series:
    numerator = (values * weights).rolling(window, center=True, min_periods=1).sum()
    denominator = weights.rolling(window, center=True, min_periods=1).sum()
    return numerator.div(denominator)


def _continuous_bin_codes(values: pd.Series, n_bins: int) -> tuple[pd.Series, pd.DataFrame]:
    """Create equal-frequency bin codes and a lookup using full-population values."""
    quantiles = values.quantile(np.linspace(0, 1, n_bins + 1)).to_numpy(dtype=float)
    edges = np.unique(quantiles[np.isfinite(quantiles)])
    if len(edges) < 3:
        raise ValueError("Feature has too few distinct values for continuous quantile bins.")
    edges[0] = np.nextafter(edges[0], -np.inf)
    edges[-1] = np.nextafter(edges[-1], np.inf)
    codes = pd.cut(values, bins=edges, labels=False, include_lowest=True)
    return codes, pd.DataFrame({"bin": range(len(edges) - 1), "bin_lower": edges[:-1], "bin_upper": edges[1:]})


def build_feature_curve(
    values: pd.Series,
    label: pd.Series,
    shap_sample_values: pd.Series,
    shap_sample_contribution: pd.Series,
    *,
    n_bins: int,
    prior_strength: float = 100.0,
) -> tuple[pd.DataFrame, str]:
    """Build full-population default-frequency and sampled-SHAP bin statistics."""
    full = pd.DataFrame({"value": pd.to_numeric(values, errors="coerce"), "label": label}).dropna()
    full = full[np.isfinite(full["value"])].copy()
    sample = pd.DataFrame(
        {
            "value": pd.to_numeric(shap_sample_values, errors="coerce"),
            "shap": shap_sample_contribution,
        }
    ).dropna()
    sample = sample[np.isfinite(sample["value"])].copy()
    if full.empty or sample.empty:
        raise ValueError("Feature has no finite values for curve construction.")

    unique_values = sorted(full["value"].unique())
    discrete = len(unique_values) <= 10
    if discrete:
        full["bin"] = full["value"]
        sample["bin"] = sample["value"]
        curve_type = "discrete"
    else:
        full_codes, lookup = _continuous_bin_codes(full["value"], n_bins)
        full["bin"] = full_codes.astype(int)
        sample["bin"] = pd.cut(
            sample["value"],
            bins=np.r_[lookup["bin_lower"].iloc[0], lookup["bin_upper"].to_numpy()],
            labels=False,
            include_lowest=True,
        ).astype(int)
        curve_type = "quantile"

    summary = (
        full.groupby("bin", observed=True)
        .agg(
            n_obs=("label", "size"),
            n_defaults=("label", "sum"),
            observed_default_rate=("label", "mean"),
            feature_value_min=("value", "min"),
            feature_value_median=("value", "median"),
            feature_value_max=("value", "max"),
        )
        .reset_index()
    )
    shap_summary = (
        sample.groupby("bin", observed=True)
        .agg(shap_sample_rows=("shap", "size"), mean_shap=("shap", "mean"))
        .reset_index()
    )
    curve = summary.merge(shap_summary, on="bin", how="left")
    curve = curve.sort_values("bin").reset_index(drop=True)
    lower, upper = _wilson_interval(curve["n_defaults"], curve["n_obs"])
    curve["default_rate_ci_lower"] = lower
    curve["default_rate_ci_upper"] = upper
    global_default_rate = full["label"].mean()
    curve["shrunk_default_rate"] = (
        curve["n_defaults"] + prior_strength * global_default_rate
    ).div(curve["n_obs"] + prior_strength)
    if curve_type == "quantile":
        curve["percentile_start"] = curve["n_obs"].cumsum().shift(fill_value=0).div(len(full)).mul(100)
        curve["percentile_end"] = curve["n_obs"].cumsum().div(len(full)).mul(100)
        curve["percentile_midpoint"] = (curve["percentile_start"] + curve["percentile_end"]) / 2
        curve["smoothed_default_rate"] = _rolling_weighted_mean(
            curve["shrunk_default_rate"], curve["n_obs"]
        )
        curve["smoothed_mean_shap"] = _rolling_weighted_mean(
            curve["mean_shap"].fillna(0), curve["shap_sample_rows"].fillna(0)
        )
    else:
        curve["percentile_start"] = np.nan
        curve["percentile_end"] = np.nan
        curve["percentile_midpoint"] = np.arange(len(curve), dtype=float)
        curve["smoothed_default_rate"] = curve["shrunk_default_rate"]
        curve["smoothed_mean_shap"] = curve["mean_shap"]
    return curve, curve_type


def save_feature_curve(
    curve: pd.DataFrame,
    *,
    feature: str,
    group: str,
    curve_type: str,
    output_path: Path,
) -> None:
    """Save a two-panel observed-default-frequency and SHAP-impact chart."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    figure, (default_axis, shap_axis) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    if curve_type == "quantile":
        x = curve["percentile_midpoint"]
        x_label = f"{feature} percentile (low to high)"
    else:
        x = np.arange(len(curve))
        x_label = feature

    default_axis.scatter(x, curve["observed_default_rate"], color="#6C757D", s=22, label="Observed bin rate")
    default_axis.plot(x, curve["smoothed_default_rate"], color="#0072B2", linewidth=2.2, label="Smoothed rate")
    default_axis.fill_between(
        x,
        curve["default_rate_ci_lower"],
        curve["default_rate_ci_upper"],
        color="#0072B2",
        alpha=0.12,
        label="95% bin CI",
    )
    default_axis.axhline(
        np.average(curve["observed_default_rate"], weights=curve["n_obs"]),
        color="0.4",
        linestyle="--",
        linewidth=1,
        label="Population default rate",
    )
    default_axis.set_ylabel("Observed 12-month default rate")
    default_axis.set_title(f"{feature} — {group.replace('_', ' ').title()}")
    default_axis.legend(loc="best", fontsize=8)
    default_axis.grid(alpha=0.2)

    shap_axis.scatter(x, curve["mean_shap"], color="#6C757D", s=22, label="Mean bin SHAP")
    shap_axis.plot(x, curve["smoothed_mean_shap"], color="#D55E00", linewidth=2.2, label="Smoothed mean SHAP")
    shap_axis.axhline(0, color="0.4", linestyle="--", linewidth=1)
    shap_axis.set(xlabel=x_label, ylabel="SHAP contribution to log-odds")
    shap_axis.legend(loc="best", fontsize=8)
    shap_axis.grid(alpha=0.2)
    if curve_type == "discrete":
        labels = [f"{value:g}" for value in curve["feature_value_median"]]
        shap_axis.set_xticks(x, labels)

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def write_report(
    selected: pd.DataFrame,
    *,
    output_path: Path,
    sample_rows: int,
    full_rows: int,
    n_bins: int,
) -> None:
    lines = [
        "# Production LightGBM feature interpretation",
        "",
        "## Method",
        "",
        f"TreeSHAP global importance was calculated on a stratified {sample_rows:,}-row sample ",
        "that preserves fiscal-year and default/non-default proportions.  The selected feature curves use ",
        f"all {full_rows:,} labeled firm-year rows: continuous features are ranked into up to {n_bins} ",
        "equal-frequency bins, then observed 12-month default rates are shrunk slightly toward the population ",
        "rate and smoothed with a centered five-bin, exposure-weighted rolling mean.",
        "",
        "This follows the RiskCalc-style default-frequency-curve idea, but the horizon here is 12 months. ",
        "SHAP shows how the fitted model uses a feature; observed bin default rates are descriptive associations, not causal effects.",
        "",
        "## Selected features",
        "",
        "| Group | Rank | Feature | Mean absolute SHAP | Curve |",
        "| --- | ---: | --- | ---: | --- |",
    ]
    for row in selected.itertuples(index=False):
        stem = str(row.feature).replace("/", "_")
        lines.append(
            f"| {str(row.group).replace('_', ' ').title()} | {row.selection_rank} | `{row.feature}` | "
            f"{row.mean_abs_shap:.5f} | [CSV](curves/{stem}.csv) · [PNG](curves/{stem}.png) |"
        )
    lines.extend(
        [
            "",
            "## Reading the charts",
            "",
            "The top panel shows each bin's observed default rate, its 95% binomial confidence interval, and the smoothed default-frequency curve.  ",
            "The bottom panel shows the corresponding mean SHAP contribution. Positive SHAP values increase the model's raw log-odds of default; negative values decrease them.",
            "",
            "One-hot legal-structure and city families are retained in `grouped_shap_importance.csv` as grouped fields, but are not selected for ratio-style curves.  High-collinearity feature pairs should be interpreted jointly.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/processed/labeled_data.csv")
    parser.add_argument("--production-dir", default="model_artifacts/production_lightgbm_v1")
    parser.add_argument(
        "--output-dir", default="reports/feature_interpretation/production_lightgbm_v1"
    )
    parser.add_argument("--sample-size", type=int, default=100_000)
    parser.add_argument("--top-features", type=int, choices=[1, 2], default=2)
    parser.add_argument("--bins", type=int, default=50)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preprocessor, model, _ = _load_production_bundle(args.production_dir)
    config = preprocessor.config
    raw = _read_labeled_data(args.input, config.target_column)
    history = raw.drop(columns=config.target_column, errors="ignore")
    transformed = transform_with_preprocessor(raw, history, preprocessor)
    feature_columns = list(model.feature_columns)
    missing = sorted(set(feature_columns).difference(transformed.columns))
    if missing:
        raise ValueError(f"Production transformation is missing model features: {missing}")
    features = transformed.reindex(columns=feature_columns).apply(pd.to_numeric, errors="coerce")
    labels = pd.to_numeric(transformed[config.target_column], errors="raise").astype(int)

    sample_positions = _stratified_sample_positions(
        transformed,
        year_column=config.year_column,
        target_column=config.target_column,
        sample_size=args.sample_size,
        random_state=args.random_state,
    )
    sample_features = features.iloc[sample_positions].copy()
    shap_values = _tree_shap_values(model.estimator, sample_features)
    importance = grouped_shap_importance(sample_features, shap_values)
    selected = select_features(importance, args.top_features)

    output_dir = project_path(args.output_dir)
    curves_dir = output_dir / "curves"
    output_dir.mkdir(parents=True, exist_ok=True)
    curves_dir.mkdir(parents=True, exist_ok=True)
    importance.to_csv(output_dir / "grouped_shap_importance.csv", index=False)
    selected.to_csv(output_dir / "selected_features.csv", index=False)

    position_by_feature = {column: position for position, column in enumerate(feature_columns)}
    for row in selected.itertuples(index=False):
        feature = str(row.feature)
        position = position_by_feature[feature]
        curve, curve_type = build_feature_curve(
            features[feature],
            labels,
            sample_features[feature],
            pd.Series(shap_values[:, position], index=sample_features.index),
            n_bins=args.bins,
        )
        curve.insert(0, "group", row.group)
        curve.insert(1, "feature", feature)
        curve.insert(2, "curve_type", curve_type)
        stem = feature.replace("/", "_")
        curve.to_csv(curves_dir / f"{stem}.csv", index=False)
        save_feature_curve(
            curve,
            feature=feature,
            group=str(row.group),
            curve_type=curve_type,
            output_path=curves_dir / f"{stem}.png",
        )

    write_report(
        selected,
        output_path=output_dir / "feature_interpretation_report.md",
        sample_rows=len(sample_features),
        full_rows=len(transformed),
        n_bins=args.bins,
    )
    print(f"Wrote grouped SHAP importance: {output_dir / 'grouped_shap_importance.csv'}")
    print(f"Wrote selected features: {output_dir / 'selected_features.csv'}")
    print(f"Wrote feature curves: {curves_dir}")
    print(f"Wrote interpretation report: {output_dir / 'feature_interpretation_report.md'}")


if __name__ == "__main__":
    main()
