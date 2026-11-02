"""Fit and apply leakage-safe cleaning, feature engineering, and history aggregation.

Each input row is a prediction anchor identified by ``(id, fs_year)``. The
output has one row per anchor, with financial features aggregated from up to
four statements for that company whose fiscal year is no later than the
anchor. The weight for a statement ``k`` calendar years old is ``phi ** k``;
available weights are re-normalized for every feature.

Use ``fit`` on a round's train.csv only. It learns sector-median imputation,
industry target encoding, and legal-structure categories, writes those
artifacts, and transforms train.csv. Use ``transform`` with that artifact for
the corresponding validation.csv and test.csv. Both commands require the
round's history.csv, but labels in history are never used as input features.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

from finance_project.common import project_path, write_json


DEBT_DETAIL = ["debt_bank_st", "debt_bank_lt", "debt_fin_st", "debt_fin_lt", "AP_st", "AP_lt"]
NEG_BS_COLS = [
    "asst_intang_fixed",
    "asst_tang_fixed",
    "asst_fixed_fin",
    "AR",
    "liab_lt",
    "liab_lt_emp",
    *DEBT_DETAIL,
    "debt_st",
    "debt_lt",
]
ANOMALY_FLAGS = [
    "neg_bs_flag",
    "neg_pl_flag",
    "fin_adj_flag",
    "neg_goodwill_flag",
    "debt_detail_inconsistent",
]
DROP_COLS = ["Unnamed: 0", "eqty_corp_family_tot", "days_rec", "roe"]
IMPUTE_COLS = [
    "asst_intang_fixed",
    "asst_tang_fixed",
    "asst_fixed_fin",
    "asst_current",
    "AR",
    "cash_and_equiv",
    "goodwill",
    "eqty_tot",
    "liab_lt",
    "liab_lt_emp",
    *DEBT_DETAIL,
    "debt_st",
    "debt_lt",
    "rev_operating",
    "COGS",
    "prof_operations",
    "inc_financing",
    "exp_financing",
    "prof_financing",
    "inc_extraord",
    "taxes",
    "profit",
    "ebitda",
    "cf_operations",
]
REQUIRED_COLUMNS = {
    "id",
    "fs_year",
    "HQ_city",
    "legal_struct",
    "ateco_sector",
    "asst_tot",
    "wc_net",
    "margin_fin",
    *IMPUTE_COLS,
}


@dataclass(frozen=True)
class PreprocessingConfig:
    """Fixed feature-engineering settings stored with fitted artifacts."""

    id_column: str = "id"
    year_column: str = "fs_year"
    target_column: str = "label"
    sector_column: str = "ateco_sector"
    legal_column: str = "legal_struct"
    phi: float = 0.5
    max_history_records: int = 4
    target_encoding_smoothing: float = 20.0
    target_encoding_folds: int = 5
    random_state: int = 42


@dataclass
class PreprocessingArtifacts:
    """Parameters learned exclusively from one round's training anchors."""

    config: PreprocessingConfig
    sector_medians: pd.DataFrame
    industry_mapping: pd.Series
    industry_global_mean: float
    legal_categories: list[str]
    feature_columns: list[str]


def _drop_index_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = [str(column).lstrip("\ufeff").strip() for column in frame.columns]
    index_columns = [
        column for column in frame.columns if not column or column.lower().startswith("unnamed")
    ]
    return frame.drop(columns=index_columns, errors="ignore")


def _require_columns(frame: pd.DataFrame, *, include_target: bool, config: PreprocessingConfig) -> None:
    required = set(REQUIRED_COLUMNS)
    if include_target:
        required.add(config.target_column)
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")


def _coerce_raw_types(frame: pd.DataFrame, config: PreprocessingConfig) -> pd.DataFrame:
    """Convert financial fields and known numeric identifiers; preserve dates/categories."""
    frame = _drop_index_columns(frame)
    numeric = set(IMPUTE_COLS) | {
        config.id_column,
        config.year_column,
        config.sector_column,
        config.target_column,
        "asst_tot",
        "HQ_city",
        "wc_net",
        "margin_fin",
        "roa",
    }
    for column in numeric.intersection(frame.columns):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def flag_anomalies(frame: pd.DataFrame) -> pd.DataFrame:
    """Add binary flags for economically implausible reported values without changing them."""
    columns = [
        *NEG_BS_COLS,
        "rev_operating",
        "COGS",
        "exp_financing",
        "ebitda",
        "prof_operations",
        "prof_financing",
        "goodwill",
    ]
    values = frame[columns].apply(pd.to_numeric, errors="coerce")
    result = frame.copy()
    result["neg_bs_flag"] = (values[NEG_BS_COLS] < 0).any(axis=1)
    result["neg_pl_flag"] = (
        (values["rev_operating"] < 0)
        | (values["COGS"] < 0)
        | (values["exp_financing"] < 0)
        | (values["ebitda"] < values["prof_operations"])
    )
    result["fin_adj_flag"] = values["prof_financing"].notna() & (values["prof_financing"] != 0)
    result["neg_goodwill_flag"] = values["goodwill"] < 0
    short_term_detail = values["debt_bank_st"] + values["debt_fin_st"] + values["AP_st"]
    long_term_detail = values["debt_bank_lt"] + values["debt_fin_lt"] + values["AP_lt"]
    result["debt_detail_inconsistent"] = values[DEBT_DETAIL].notna().all(axis=1) & (
        (short_term_detail > values["debt_st"] + 1)
        | (long_term_detail > values["debt_lt"] + 1)
    )
    result[ANOMALY_FLAGS] = result[ANOMALY_FLAGS].astype(int)
    return result


def fit_sector_medians(train: pd.DataFrame, config: PreprocessingConfig) -> pd.DataFrame:
    """Fit sector-level size-normalized imputation medians on training anchors only."""
    train = _coerce_raw_types(train, config)
    ratios = train[IMPUTE_COLS].div(train["asst_tot"], axis=0)
    ratios["asst_tot"] = train["asst_tot"]
    medians = ratios.groupby(train[config.sector_column]).median()
    medians.loc["ALL"] = ratios.median()
    return medians


def clean_and_impute(
    frame: pd.DataFrame,
    medians: pd.DataFrame,
    config: PreprocessingConfig,
) -> pd.DataFrame:
    """Flag anomalies, preserve their values, and impute raw fields with training statistics."""
    _require_columns(frame, include_target=False, config=config)
    result = _coerce_raw_types(frame, config)
    result = flag_anomalies(result)
    result = result.drop(columns=DROP_COLS, errors="ignore")

    result["HQ_city"] = (
        result["HQ_city"].map("{:.0f}".format, na_action="ignore").fillna("Unknown")
    )
    result[config.legal_column] = result[config.legal_column].fillna("Unknown").astype(str)
    result["debt_detail_missing"] = result[DEBT_DETAIL].isna().any(axis=1).astype(int)

    # Restore a missing component only when the other two income-statement
    # values were reported. This happens before ratio-based imputation.
    result["rev_operating"] = result["rev_operating"].fillna(
        result["prof_operations"] + result["COGS"]
    )
    result["COGS"] = result["COGS"].fillna(
        result["rev_operating"] - result["prof_operations"]
    )
    result["prof_operations"] = result["prof_operations"].fillna(
        result["rev_operating"] - result["COGS"]
    )

    by_sector = medians.drop(index="ALL").reindex(result[config.sector_column])
    by_sector.index = result.index
    by_sector = by_sector.fillna(medians.loc["ALL"])
    result["asst_tot"] = result["asst_tot"].fillna(by_sector["asst_tot"])
    result[IMPUTE_COLS] = result[IMPUTE_COLS].fillna(
        by_sector[IMPUTE_COLS].mul(result["asst_tot"], axis=0)
    )

    fixed_assets = (
        result["asst_intang_fixed"] + result["asst_tang_fixed"] + result["asst_fixed_fin"]
    )
    result["margin_fin"] = result["margin_fin"].fillna(result["eqty_tot"] - fixed_assets)
    result["wc_net"] = result["wc_net"].fillna(result["asst_current"] - result["debt_st"])
    result["roa"] = result.get("roa", pd.Series(index=result.index, dtype=float)).fillna(
        result["prof_operations"].div(result["asst_tot"]).mul(100)
    )
    return result


def _positive_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Return NaN rather than infinity when a financial denominator is non-positive."""
    return numerator.div(denominator.where(denominator > 0))


def make_period_features(frame: pd.DataFrame, config: PreprocessingConfig) -> pd.DataFrame:
    """Create financial ratios and flags for each firm-year before history aggregation."""
    result = frame.sort_values([config.id_column, config.year_column]).copy()
    total_debt = result["debt_st"] + result["debt_lt"]
    bank_debt = result["debt_bank_st"] + result["debt_bank_lt"]
    long_term_capital = result["debt_lt"] + result["eqty_tot"]

    result["size"] = np.log(result["asst_tot"].where(result["asst_tot"] > 0))
    result["st_debt_share"] = _positive_divide(result["debt_st"], total_debt)
    result["lt_debt_to_capital"] = _positive_divide(result["debt_lt"], long_term_capital)
    result["bank_debt_conc"] = _positive_divide(bank_debt, total_debt)
    result["cash_ratio"] = _positive_divide(result["cash_and_equiv"], result["asst_tot"])
    result["cash_to_financing_exp_ratio"] = _positive_divide(
        result["cash_and_equiv"], result["exp_financing"]
    )
    result["cur_asst_ratio"] = _positive_divide(result["asst_current"], result["asst_tot"])
    result["AR_ratio"] = _positive_divide(result["AR"], result["rev_operating"])
    result["payables_ratio"] = _positive_divide(result["AP_st"], result["asst_tot"])
    result["wc_ratio"] = _positive_divide(result["wc_net"], result["asst_tot"])
    result["cur_debt_ratio"] = _positive_divide(result["asst_current"], result["debt_st"])
    result["ebitda_ratio"] = _positive_divide(result["ebitda"], result["asst_tot"])
    result["op_margin"] = _positive_divide(result["prof_operations"], result["rev_operating"])
    result["net_prof_margin"] = _positive_divide(result["profit"], result["rev_operating"])
    result["int_coverage"] = _positive_divide(result["ebitda"], result["exp_financing"])
    result["cash_to_debt"] = _positive_divide(result["cf_operations"], total_debt)
    result["debt_to_ebitda"] = _positive_divide(total_debt, result["ebitda"])
    result["loss_flag"] = (result["profit"] < 0).astype(int)

    previous_year = result.groupby(config.id_column)[config.year_column].shift()
    year_gap = result[config.year_column] - previous_year
    previous_roa = result.groupby(config.id_column)["roa"].shift()
    result["annualized_change_in_roa"] = (
        (result["roa"] - previous_roa).div(year_gap.where(year_gap > 0))
    )

    previous_revenue = result.groupby(config.id_column)["rev_operating"].shift()
    revenue_ratio = _positive_divide(result["rev_operating"], previous_revenue)
    result["annualized_sales_growth"] = (
        revenue_ratio.pow(1 / year_gap.where(year_gap > 0)) - 1
    ).where((result["rev_operating"] > 0) & (previous_revenue > 0))

    previous_ar = result.groupby(config.id_column)["AR"].shift()
    average_ar = (result["AR"] + previous_ar) / 2
    result["ar_turnover"] = _positive_divide(result["rev_operating"], average_ar)
    previous_ar_turnover = result.groupby(config.id_column)["ar_turnover"].shift()
    result["change_in_ar_turnover"] = (
        (result["ar_turnover"] - previous_ar_turnover).div(year_gap.where(year_gap > 0))
    )

    zero_values = {
        "zero_asst_tot": result["asst_tot"],
        "zero_total_debt": total_debt,
        "zero_debt_st": result["debt_st"],
        "zero_debt_lt": result["debt_lt"],
        "zero_bank_debt": bank_debt,
        "zero_cash": result["cash_and_equiv"],
        "zero_asst_current": result["asst_current"],
        "zero_ar": result["AR"],
        "zero_rev_operating": result["rev_operating"],
        "zero_AP_st": result["AP_st"],
        "zero_wc_net": result["wc_net"],
        "zero_ebitda": result["ebitda"],
        "zero_prof_operations": result["prof_operations"],
        "zero_profit": result["profit"],
        "zero_cf_operations": result["cf_operations"],
        "zero_exp_financing": result["exp_financing"],
    }
    for name, values in zero_values.items():
        result[name] = values.eq(0).astype(int)
    return result


def _period_numeric_features() -> list[str]:
    return [
        "size",
        "st_debt_share",
        "lt_debt_to_capital",
        "bank_debt_conc",
        "cash_ratio",
        "cash_to_financing_exp_ratio",
        "cur_asst_ratio",
        "AR_ratio",
        "change_in_ar_turnover",
        "payables_ratio",
        "wc_ratio",
        "cur_debt_ratio",
        "annualized_change_in_roa",
        "ebitda_ratio",
        "op_margin",
        "annualized_sales_growth",
        "net_prof_margin",
        "int_coverage",
        "cash_to_debt",
        "debt_to_ebitda",
    ]


def _period_binary_features() -> list[str]:
    return [
        *ANOMALY_FLAGS,
        "debt_detail_missing",
        "loss_flag",
        "zero_asst_tot",
        "zero_total_debt",
        "zero_debt_st",
        "zero_debt_lt",
        "zero_bank_debt",
        "zero_cash",
        "zero_asst_current",
        "zero_ar",
        "zero_rev_operating",
        "zero_AP_st",
        "zero_wc_net",
        "zero_ebitda",
        "zero_prof_operations",
        "zero_profit",
        "zero_cf_operations",
        "zero_exp_financing",
    ]


def _assert_anchor_history_compatibility(
    anchors: pd.DataFrame,
    history: pd.DataFrame,
    config: PreprocessingConfig,
) -> None:
    key_columns = [config.id_column, config.year_column]
    if anchors.duplicated(key_columns).any():
        raise ValueError("Input anchors must have one row per (id, fs_year).")
    history_keys = pd.MultiIndex.from_frame(history[key_columns])
    anchor_keys = pd.MultiIndex.from_frame(anchors[key_columns])
    if not anchor_keys.isin(history_keys).all():
        raise ValueError("Every input anchor must also be present in the supplied history file.")


def aggregate_histories(
    anchors: pd.DataFrame,
    history_features: pd.DataFrame,
    config: PreprocessingConfig,
) -> pd.DataFrame:
    """Aggregate up to four available prior/current statements for every anchor.

    Weights decay with the calendar-year difference, not merely with a record's
    position. Binary flags use a history maximum so they remain binary.
    """
    _assert_anchor_history_compatibility(anchors, history_features, config)
    id_column = config.id_column
    year_column = config.year_column
    anchor_columns = [id_column, year_column, config.sector_column, config.legal_column]
    for optional in [config.target_column, "pred_start_dt"]:
        if optional in anchors.columns:
            anchor_columns.append(optional)

    anchor_table = anchors[anchor_columns].copy().reset_index(drop=True)
    anchor_table["_anchor_row"] = np.arange(len(anchor_table))
    history_columns = [id_column, year_column, *_period_numeric_features(), *_period_binary_features()]
    matches = anchor_table[["_anchor_row", id_column, year_column]].merge(
        history_features[history_columns], on=id_column, how="left", suffixes=("_anchor", "_history")
    )
    matches["_age"] = matches[f"{year_column}_anchor"] - matches[f"{year_column}_history"]
    matches = matches.loc[matches["_age"] >= 0].copy()
    matches = matches.sort_values(["_anchor_row", "_age"], ascending=[True, True])
    matches = matches.groupby("_anchor_row", sort=False).head(config.max_history_records)
    matches["_weight"] = config.phi ** matches["_age"]

    output = anchor_table.set_index("_anchor_row")
    for column in _period_numeric_features():
        valid_weight = matches["_weight"].where(matches[column].notna())
        denominator = valid_weight.groupby(matches["_anchor_row"]).sum(min_count=1)
        numerator = (matches[column] * matches["_weight"]).groupby(matches["_anchor_row"]).sum(
            min_count=1
        )
        output[column] = numerator.div(denominator).reindex(output.index)
    for column in _period_binary_features():
        output[column] = (
            matches.groupby("_anchor_row")[column].max().reindex(output.index).fillna(0).astype(int)
        )
    output["history_records"] = matches.groupby("_anchor_row").size().reindex(output.index).astype(int)
    output["no_prior_observation"] = (output["history_records"] == 1).astype(int)
    return output.reset_index(drop=True)


def _oof_target_encode(
    frame: pd.DataFrame,
    config: PreprocessingConfig,
) -> tuple[pd.Series, pd.Series, float]:
    """Group-aware, out-of-fold target encode industry for training anchors."""
    target = frame[config.target_column]
    category = frame[config.sector_column]
    global_mean = float(target.mean())
    encoded = pd.Series(index=frame.index, dtype=float)
    group_count = frame[config.id_column].nunique()
    folds = min(config.target_encoding_folds, group_count)
    if folds < 2:
        raise ValueError("Target encoding requires at least two distinct training company IDs.")

    class_counts = target.value_counts()
    if target.nunique() > 1 and class_counts.min() >= folds:
        splitter: Any = StratifiedGroupKFold(
            n_splits=folds, shuffle=True, random_state=config.random_state
        )
        fold_iterator = splitter.split(frame, target, groups=frame[config.id_column])
    else:
        splitter = GroupKFold(n_splits=folds)
        fold_iterator = splitter.split(frame, groups=frame[config.id_column])

    for fit_index, validation_index in fold_iterator:
        fit = frame.iloc[fit_index]
        fold_mean = float(fit[config.target_column].mean())
        stats = fit.groupby(config.sector_column)[config.target_column].agg(["sum", "count"])
        mapping = (stats["sum"] + config.target_encoding_smoothing * fold_mean) / (
            stats["count"] + config.target_encoding_smoothing
        )
        encoded.iloc[validation_index] = category.iloc[validation_index].map(mapping).fillna(fold_mean)

    full_stats = frame.groupby(config.sector_column)[config.target_column].agg(["sum", "count"])
    full_mapping = (full_stats["sum"] + config.target_encoding_smoothing * global_mean) / (
        full_stats["count"] + config.target_encoding_smoothing
    )
    return encoded, full_mapping, global_mean


def _finalize_features(
    aggregated: pd.DataFrame,
    artifacts: PreprocessingArtifacts,
    *,
    training: bool,
) -> pd.DataFrame:
    config = artifacts.config
    output = aggregated.copy()
    if training:
        encoded, _, _ = _oof_target_encode(output, config)
        output["ateco_sector_te"] = encoded
    else:
        output["ateco_sector_te"] = output[config.sector_column].map(
            artifacts.industry_mapping
        ).fillna(artifacts.industry_global_mean)

    legal = output[config.legal_column].fillna("Unknown").astype(str)
    for category in artifacts.legal_categories:
        output[f"legal_{category}"] = legal.eq(category).astype(int)

    metadata = [config.id_column, config.year_column]
    for optional in ["pred_start_dt", config.target_column]:
        if optional in output.columns:
            metadata.append(optional)
    ordered_features = artifacts.feature_columns
    return output[[*metadata, *ordered_features]]


def _build_aggregated(
    anchors: pd.DataFrame,
    history: pd.DataFrame,
    medians: pd.DataFrame,
    config: PreprocessingConfig,
) -> pd.DataFrame:
    clean_anchors = clean_and_impute(anchors, medians, config)
    clean_history = clean_and_impute(history, medians, config)
    return aggregate_histories(clean_anchors, make_period_features(clean_history, config), config)


def fit_preprocessor(
    train: pd.DataFrame,
    history: pd.DataFrame,
    config: PreprocessingConfig = PreprocessingConfig(),
) -> tuple[pd.DataFrame, PreprocessingArtifacts]:
    """Fit all learned transformations on train only and return transformed train anchors."""
    train = _drop_index_columns(train)
    history = _drop_index_columns(history)
    _require_columns(train, include_target=True, config=config)
    _require_columns(history, include_target=False, config=config)
    medians = fit_sector_medians(train, config)
    aggregated = _build_aggregated(train, history, medians, config)
    _, industry_mapping, industry_global_mean = _oof_target_encode(aggregated, config)
    legal_categories = sorted(aggregated[config.legal_column].fillna("Unknown").astype(str).unique())
    feature_columns = [
        *_period_numeric_features(),
        *_period_binary_features(),
        "history_records",
        "no_prior_observation",
        "ateco_sector_te",
        *[f"legal_{category}" for category in legal_categories],
    ]
    artifacts = PreprocessingArtifacts(
        config=config,
        sector_medians=medians,
        industry_mapping=industry_mapping,
        industry_global_mean=industry_global_mean,
        legal_categories=legal_categories,
        feature_columns=feature_columns,
    )
    return _finalize_features(aggregated, artifacts, training=True), artifacts


def transform_with_preprocessor(
    anchors: pd.DataFrame,
    history: pd.DataFrame,
    artifacts: PreprocessingArtifacts,
) -> pd.DataFrame:
    """Transform validation, test, or holdout anchors without refitting any statistics."""
    config = artifacts.config
    anchors = _drop_index_columns(anchors)
    history = _drop_index_columns(history)
    _require_columns(anchors, include_target=False, config=config)
    _require_columns(history, include_target=False, config=config)
    aggregated = _build_aggregated(anchors, history, artifacts.sector_medians, config)
    return _finalize_features(aggregated, artifacts, training=False)


def _read_csv(path: str) -> pd.DataFrame:
    source = project_path(path)
    if not source.exists():
        raise FileNotFoundError(f"Input data not found: {source}")
    return pd.read_csv(source, low_memory=False)


def _write_csv(frame: pd.DataFrame, path: str) -> Path:
    destination = project_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    return destination


def _fit_command(args: argparse.Namespace) -> None:
    config = PreprocessingConfig(
        id_column=args.id_column,
        year_column=args.year_column,
        target_column=args.target_column,
        sector_column=args.sector_column,
        legal_column=args.legal_column,
        phi=args.phi,
        max_history_records=args.max_history_records,
        target_encoding_smoothing=args.target_encoding_smoothing,
        target_encoding_folds=args.target_encoding_folds,
        random_state=args.random_state,
    )
    transformed, artifacts = fit_preprocessor(_read_csv(args.input), _read_csv(args.history), config)
    output_path = _write_csv(transformed, args.output)
    artifact_path = project_path(args.artifacts)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifacts, artifact_path)
    summary = write_json(
        args.summary_output,
        {
            "mode": "fit",
            "input": str(project_path(args.input)),
            "history": str(project_path(args.history)),
            "output": str(output_path),
            "artifacts": str(artifact_path),
            "rows": int(len(transformed)),
            "features": artifacts.feature_columns,
            "config": asdict(config),
        },
    )
    print(f"Wrote transformed training features: {output_path}")
    print(f"Wrote preprocessing artifacts: {artifact_path}")
    print(f"Wrote preprocessing summary: {summary}")


def _transform_command(args: argparse.Namespace) -> None:
    artifact_path = project_path(args.artifacts)
    if not artifact_path.exists():
        raise FileNotFoundError(f"Preprocessing artifacts not found: {artifact_path}")
    artifacts = joblib.load(artifact_path)
    if not isinstance(artifacts, PreprocessingArtifacts):
        raise TypeError("Artifact is not a PreprocessingArtifacts object.")
    transformed = transform_with_preprocessor(_read_csv(args.input), _read_csv(args.history), artifacts)
    output_path = _write_csv(transformed, args.output)
    print(f"Wrote transformed features: {output_path}")


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", required=True, help="Anchor CSV: train.csv, validation.csv, or test.csv.")
    parser.add_argument("--history", required=True, help="Round-specific history.csv from make_dataset.")
    parser.add_argument("--output", required=True, help="Transformed feature CSV output path.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("fit", help="Fit preprocessing on one round's training anchors.")
    _add_common_arguments(fit)
    fit.add_argument("--artifacts", required=True, help="Joblib artifact output path.")
    fit.add_argument("--summary-output", default="reports/metrics/preprocessing_fit.json")
    fit.add_argument("--id-column", default="id")
    fit.add_argument("--year-column", default="fs_year")
    fit.add_argument("--target-column", default="label")
    fit.add_argument("--sector-column", default="ateco_sector")
    fit.add_argument("--legal-column", default="legal_struct")
    fit.add_argument("--phi", type=float, default=0.5)
    fit.add_argument("--max-history-records", type=int, default=4)
    fit.add_argument("--target-encoding-smoothing", type=float, default=20.0)
    fit.add_argument("--target-encoding-folds", type=int, default=5)
    fit.add_argument("--random-state", type=int, default=42)
    transform = commands.add_parser("transform", help="Transform validation/test data with fitted artifacts.")
    _add_common_arguments(transform)
    transform.add_argument("--artifacts", required=True, help="Joblib artifact input path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "fit":
        _fit_command(args)
    else:
        _transform_command(args)


if __name__ == "__main__":
    main()
