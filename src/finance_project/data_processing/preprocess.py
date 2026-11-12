"""Fit and apply leakage-safe cleaning and feature engineering.

Each input row is a prediction anchor identified by ``(id, fs_year)``. The
output has one row per anchor. ``decayed_history`` aggregates financial
features from up to four statements whose fiscal year is no later than the
anchor. ``firm_year`` retains only the anchor-year features, while still using
earlier statements to calculate backward-looking change features. The weight
for a statement ``k`` calendar years old is ``phi ** k``; available weights
are re-normalized for every feature in ``decayed_history`` mode.

Use ``fit`` on a round's train.csv only. It learns sector-median imputation,
industry target encoding, and legal-structure categories, writes those
artifacts, and transforms train.csv. Use ``transform`` with that artifact for
the corresponding validation.csv and test.csv. Both commands require the
round's history.csv, but labels in history are never used as input features.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

from finance_project.common import project_path, write_json


DEBT_DETAIL = ["debt_bank_st", "debt_bank_lt", "debt_fin_st", "debt_fin_lt", "AP_st", "AP_lt"]
ANOMALY_FLAGS = ["neg_bs_flag", "neg_pl_flag", "debt_detail_inconsistent"]
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
    feature_mode: str = "decayed_history"
    target_encoding_smoothing: float = 20.0
    target_encoding_folds: int = 5
    random_state: int = 42

    def __post_init__(self) -> None:
        if self.feature_mode not in {"decayed_history", "firm_year"}:
            raise ValueError("feature_mode must be 'decayed_history' or 'firm_year'.")


@dataclass
class PreprocessingArtifacts:
    """Parameters learned exclusively from one round's training anchors."""

    config: PreprocessingConfig
    sector_medians: pd.DataFrame
    industry_mapping: pd.Series
    industry_global_mean: float
    legal_categories: list[str]
    legal_reference_category: str
    city_categories: list[str]
    city_reference_category: str
    feature_columns: list[str]


def _load_notebook_functions(notebook: str) -> dict[str, Any]:
    """Exec the code cells tagged ``export`` in notebooks/features/<notebook>.ipynb."""
    cells = json.loads(project_path(f"notebooks/features/{notebook}.ipynb").read_text(encoding="utf-8"))["cells"]
    namespace: dict[str, Any] = {"pd": pd, "np": np}
    for cell in cells:
        if "export" in cell["metadata"].get("tags", []):
            exec("".join(cell["source"]), namespace)
    return namespace


_flags = _load_notebook_functions("01_flag_anomalies")
_missing = _load_notebook_functions("02_missing_values")
flag_anomalies = _flags["flag_anomalies"]
fill_missing = _missing["fill_missing"]


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


def clean_and_impute(
    frame: pd.DataFrame,
    medians: pd.DataFrame,
    config: PreprocessingConfig,
) -> pd.DataFrame:
    """Flag anomalies, preserve their values, and impute raw fields with training statistics."""
    frame = _drop_index_columns(frame)
    _require_columns(frame, include_target=False, config=config)
    result = fill_missing(flag_anomalies(frame), medians)
    result[config.legal_column] = result[config.legal_column].fillna("Unknown").astype(str)
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
    result["equity_ratio"] = _positive_divide(result["eqty_tot"], result["asst_tot"])
    result["debt_ratio"] = _positive_divide(total_debt, result["asst_tot"])
    result["st_debt_share"] = _positive_divide(result["debt_st"], total_debt)
    result["lt_debt_to_capital"] = _positive_divide(result["debt_lt"], long_term_capital)
    result["bank_debt_conc"] = _positive_divide(bank_debt, total_debt)
    result["tangible_asset_ratio"] = _positive_divide(result["asst_tang_fixed"], result["asst_tot"])
    result["goodwill_ratio"] = _positive_divide(result["goodwill"], result["asst_tot"])
    result["cash_ratio"] = _positive_divide(result["cash_and_equiv"], result["asst_tot"])
    result["cash_to_debt"] = _positive_divide(result["cash_and_equiv"], total_debt)
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
    result["tax_ratio"] = _positive_divide(result["taxes"], result["asst_tot"])
    result["cogs_ratio"] = _positive_divide(result["COGS"], result["rev_operating"])
    result["int_coverage"] = _positive_divide(result["ebitda"], result["exp_financing"])
    result["financing_expense_ratio"] = _positive_divide(result["exp_financing"], result["asst_tot"])
    result["financing_income_ratio"] = _positive_divide(result["inc_financing"], result["asst_tot"])
    result["operating_cash_flow_ratio"] = _positive_divide(result["cf_operations"], result["asst_tot"])
    result["operating_cash_flow_to_debt"] = _positive_divide(result["cf_operations"], total_debt)
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
    # The feature schema defines this as a difference in turnover, rather than
    # an annualized change. (Unlike ROA and sales growth, it is not divided by
    # the calendar-year gap.)
    result["change_in_ar_turnover"] = result["ar_turnover"] - previous_ar_turnover

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
        "equity_ratio",
        "debt_ratio",
        "st_debt_share",
        "lt_debt_to_capital",
        "bank_debt_conc",
        "tangible_asset_ratio",
        "goodwill_ratio",
        "cash_ratio",
        "cash_to_financing_exp_ratio",
        "cash_to_debt",
        "cur_asst_ratio",
        "AR_ratio",
        "change_in_ar_turnover",
        "payables_ratio",
        "wc_ratio",
        "cur_debt_ratio",
        "annualized_change_in_roa",
        "roa",
        "roe",
        "ebitda_ratio",
        "op_margin",
        "annualized_sales_growth",
        "net_prof_margin",
        "tax_ratio",
        "cogs_ratio",
        "int_coverage",
        "financing_expense_ratio",
        "financing_income_ratio",
        "operating_cash_flow_ratio",
        "operating_cash_flow_to_debt",
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


# These groups mirror data/interim/Feature_Schema-Sheet1.csv. A trailing '*'
# denotes one or more one-hot encoded columns.
FEATURE_GROUPS: dict[str, list[str]] = {
    "business_structural": ["ateco_sector_te", "legal_*", "city_*", "size"],
    "leverage_capital_structure": [
        "equity_ratio",
        "debt_ratio",
        "st_debt_share",
        "lt_debt_to_capital",
        "bank_debt_conc",
        "cur_asst_ratio",
        "tangible_asset_ratio",
        "goodwill_ratio",
        "neg_bs_flag",
    ],
    "liquidity_activity": [
        "cash_ratio",
        "cash_to_debt",
        "AR_ratio",
        "change_in_ar_turnover",
        "payables_ratio",
        "wc_ratio",
        "cur_debt_ratio",
    ],
    "profitability_growth": [
        "roa",
        "annualized_change_in_roa",
        "ebitda_ratio",
        "op_margin",
        "annualized_sales_growth",
        "net_prof_margin",
        "roe",
        "tax_ratio",
        "cogs_ratio",
        "neg_pl_flag",
    ],
    "coverage_cash_flow": [
        "int_coverage",
        "financing_expense_ratio",
        "cash_to_financing_exp_ratio",
        "debt_to_ebitda",
        "financing_income_ratio",
        "operating_cash_flow_ratio",
        "operating_cash_flow_to_debt",
    ],
}


def feature_schema_coverage(feature_columns: list[str]) -> pd.DataFrame:
    """Audit implemented model features against Feature_Schema-Sheet1.csv.

    A one-hot categorical family is considered represented when it has at
    least one non-reference dummy. A single observed category legitimately has
    no dummy because it has zero variance.
    """
    available = set(feature_columns)
    rows: list[dict[str, str | bool]] = []
    for group, requirements in FEATURE_GROUPS.items():
        for requirement in requirements:
            if requirement.endswith("*"):
                prefix = requirement[:-1]
                present = any(column.startswith(prefix) for column in available)
            else:
                present = requirement in available
            rows.append(
                {
                    "group": group,
                    "schema_feature": requirement,
                    "implemented": present,
                }
            )
    return pd.DataFrame(rows)


def _group_feature_columns(feature_columns: list[str], requirements: list[str]) -> list[str]:
    """Resolve exact and prefix feature requirements for a schema group."""
    selected: list[str] = []
    for requirement in requirements:
        if requirement.endswith("*"):
            selected.extend(column for column in feature_columns if column.startswith(requirement[:-1]))
        elif requirement in feature_columns:
            selected.append(requirement)
    return list(dict.fromkeys(selected))


def calculate_vif(
    frame: pd.DataFrame,
    feature_columns: list[str],
    *,
    max_rows: int = 100_000,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return VIF, covariance, and correlation diagnostics for numeric features.

    Missing values are median-filled only for this diagnostic. All-missing and
    constant columns are retained in the VIF table with an explanatory status
    but excluded from the matrix inversion. To keep the diagnostic practical
    on the full panel, it uses a reproducible row sample when needed.
    """
    if max_rows < 2:
        raise ValueError("max_rows must be at least 2.")
    values = frame.reindex(columns=feature_columns).apply(pd.to_numeric, errors="coerce")
    values = values.replace([np.inf, -np.inf], np.nan)
    if len(values) > max_rows:
        values = values.sample(n=max_rows, random_state=random_state)

    rows: list[dict[str, str | float | int]] = []
    eligible: list[str] = []
    for column in feature_columns:
        series = values[column]
        if series.notna().sum() == 0:
            rows.append({"feature": column, "vif": np.nan, "status": "all_missing"})
        elif series.nunique(dropna=True) < 2:
            rows.append({"feature": column, "vif": np.nan, "status": "constant"})
        else:
            eligible.append(column)

    usable = values[eligible].copy()
    if eligible:
        usable = usable.fillna(usable.median()).fillna(0.0)
    covariance = usable.cov() if eligible else pd.DataFrame()
    correlation = usable.corr() if eligible else pd.DataFrame()

    if len(eligible) == 1:
        rows.append({"feature": eligible[0], "vif": 1.0, "status": "ok"})
    elif len(eligible) > 1:
        matrix = correlation.to_numpy(dtype=float)
        rank = int(np.linalg.matrix_rank(matrix))
        if rank < len(eligible):
            rows.extend({"feature": column, "vif": np.inf, "status": "singular_matrix"} for column in eligible)
        else:
            inverse = np.linalg.inv(matrix)
            rows.extend(
                {
                    "feature": column,
                    "vif": float(inverse[position, position]),
                    "status": "ok",
                }
                for position, column in enumerate(eligible)
            )

    vif = pd.DataFrame(rows)
    if not vif.empty:
        vif.insert(1, "sample_rows", len(values))
        vif = vif.sort_values(["vif", "feature"], ascending=[False, True], na_position="last").reset_index(drop=True)
    return vif, covariance, correlation


def feature_group_collinearity(
    frame: pd.DataFrame,
    feature_columns: list[str] | None = None,
    *,
    max_rows: int = 100_000,
    random_state: int = 42,
) -> dict[str, dict[str, pd.DataFrame]]:
    """Calculate VIF-first diagnostics for every Feature Schema group.

    Pass transformed *training* features only. This avoids using validation or
    test data even for an exploratory diagnostic.
    """
    feature_columns = list(frame.columns) if feature_columns is None else feature_columns
    diagnostics: dict[str, dict[str, pd.DataFrame]] = {}
    for group, requirements in FEATURE_GROUPS.items():
        columns = _group_feature_columns(feature_columns, requirements)
        vif, covariance, correlation = calculate_vif(
            frame,
            columns,
            max_rows=max_rows,
            random_state=random_state,
        )
        diagnostics[group] = {
            "vif": vif,
            "covariance": covariance,
            "correlation": correlation,
        }
    return diagnostics


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
    anchor_columns = [
        id_column,
        year_column,
        config.sector_column,
        config.legal_column,
        "HQ_city",
    ]
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


def select_firm_year_features(
    anchors: pd.DataFrame,
    history_features: pd.DataFrame,
    config: PreprocessingConfig,
) -> pd.DataFrame:
    """Keep direct anchor-year features while retaining only backward-looking lags.

    ``make_period_features`` is evaluated on the supplied history first, so a
    change feature at fiscal year ``t`` can use only the same firm's earlier
    statement(s). This function then selects the exact ``(id, fs_year)``
    anchor row rather than decay-averaging earlier values.
    """
    _assert_anchor_history_compatibility(anchors, history_features, config)
    id_column = config.id_column
    year_column = config.year_column
    key_columns = [id_column, year_column]
    if history_features.duplicated(key_columns).any():
        raise ValueError("History must have at most one row per (id, fs_year).")

    anchor_columns = [
        id_column,
        year_column,
        config.sector_column,
        config.legal_column,
        "HQ_city",
    ]
    for optional in [config.target_column, "pred_start_dt"]:
        if optional in anchors.columns:
            anchor_columns.append(optional)
    period_columns = [*_period_numeric_features(), *_period_binary_features()]
    direct_features = history_features[[*key_columns, *period_columns]]
    return anchors[anchor_columns].merge(
        direct_features,
        on=key_columns,
        how="left",
        validate="one_to_one",
    )


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
    city = output["HQ_city"].fillna("Unknown").astype(str)
    for category in artifacts.city_categories:
        output[f"city_{category}"] = city.eq(category).astype(int)

    metadata = [config.id_column, config.year_column]
    for optional in ["pred_start_dt", config.target_column]:
        if optional in output.columns:
            metadata.append(optional)
    ordered_features = artifacts.feature_columns
    return output[[*metadata, *ordered_features]]


def _build_features(
    anchors: pd.DataFrame,
    history: pd.DataFrame,
    medians: pd.DataFrame,
    config: PreprocessingConfig,
) -> pd.DataFrame:
    clean_anchors = clean_and_impute(anchors, medians, config)
    clean_history = clean_and_impute(history, medians, config)
    history_features = make_period_features(clean_history, config)
    if config.feature_mode == "decayed_history":
        return aggregate_histories(clean_anchors, history_features, config)
    if config.feature_mode == "firm_year":
        return select_firm_year_features(clean_anchors, history_features, config)
    raise ValueError("feature_mode must be 'decayed_history' or 'firm_year'.")


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
    medians = _missing["sector_medians"](_drop_index_columns(train))
    engineered = _build_features(train, history, medians, config)
    _, industry_mapping, industry_global_mean = _oof_target_encode(engineered, config)
    all_legal_categories = sorted(engineered[config.legal_column].fillna("Unknown").astype(str).unique())
    all_city_categories = sorted(engineered["HQ_city"].fillna("Unknown").astype(str).unique())
    legal_reference_category, *legal_categories = all_legal_categories
    city_reference_category, *city_categories = all_city_categories
    feature_columns = [
        *_period_numeric_features(),
        *_period_binary_features(),
    ]
    if config.feature_mode == "decayed_history":
        feature_columns.extend(["history_records", "no_prior_observation"])
    feature_columns.extend(
        [
            "ateco_sector_te",
            *[f"legal_{category}" for category in legal_categories],
            *[f"city_{category}" for category in city_categories],
        ]
    )
    schema_coverage = feature_schema_coverage(feature_columns)
    missing_required = schema_coverage.loc[
        ~schema_coverage["implemented"] & ~schema_coverage["schema_feature"].str.endswith("*"),
        "schema_feature",
    ].tolist()
    if missing_required:
        raise AssertionError(f"Feature Schema fields are missing: {missing_required}")
    artifacts = PreprocessingArtifacts(
        config=config,
        sector_medians=medians,
        industry_mapping=industry_mapping,
        industry_global_mean=industry_global_mean,
        legal_categories=legal_categories,
        legal_reference_category=legal_reference_category,
        city_categories=city_categories,
        city_reference_category=city_reference_category,
        feature_columns=feature_columns,
    )
    return _finalize_features(engineered, artifacts, training=True), artifacts


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
    engineered = _build_features(anchors, history, artifacts.sector_medians, config)
    return _finalize_features(engineered, artifacts, training=False)


def bootstrap_minority_class(
    frame: pd.DataFrame,
    *,
    target_column: str = "label",
    minority_to_majority_ratio: float = 0.1,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict[str, int | float | bool]]:
    """Bootstrap the minority class after feature construction.

    ``minority_to_majority_ratio=0.1`` targets one minority sample for every
    ten majority samples; ``0.2`` targets one for every five. The function
    never changes validation or test data and never refits preprocessing.
    """
    if not 0 < minority_to_majority_ratio <= 1:
        raise ValueError("minority_to_majority_ratio must be in the interval (0, 1].")
    if target_column not in frame.columns:
        raise ValueError(f"Target column '{target_column}' is not present for bootstrapping.")
    if frame[target_column].isna().any():
        raise ValueError("Bootstrapping requires a non-missing training target.")

    counts = frame[target_column].value_counts()
    if len(counts) != 2:
        raise ValueError("Bootstrapping is supported only for a binary target.")
    minority_label = counts.idxmin()
    majority_label = counts.idxmax()
    minority = frame.loc[frame[target_column] == minority_label]
    majority = frame.loc[frame[target_column] == majority_label]
    target_minority_rows = int(np.ceil(len(majority) * minority_to_majority_ratio))

    if len(minority) >= target_minority_rows:
        return frame.sample(frac=1, random_state=random_state).reset_index(drop=True), {
            "applied": False,
            "minority_label": minority_label,
            "majority_label": majority_label,
            "minority_rows_before": int(len(minority)),
            "minority_rows_after": int(len(minority)),
            "majority_rows": int(len(majority)),
            "minority_to_majority_ratio": minority_to_majority_ratio,
        }

    bootstrapped_minority = minority.sample(
        n=target_minority_rows,
        replace=True,
        random_state=random_state,
    )
    resampled = pd.concat([majority, bootstrapped_minority], ignore_index=True)
    resampled = resampled.sample(frac=1, random_state=random_state).reset_index(drop=True)
    return resampled, {
        "applied": True,
        "minority_label": minority_label,
        "majority_label": majority_label,
        "minority_rows_before": int(len(minority)),
        "minority_rows_after": int(len(bootstrapped_minority)),
        "majority_rows": int(len(majority)),
        "minority_to_majority_ratio": minority_to_majority_ratio,
    }


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
        feature_mode=args.feature_mode,
        target_encoding_smoothing=args.target_encoding_smoothing,
        target_encoding_folds=args.target_encoding_folds,
        random_state=args.random_state,
    )
    transformed, artifacts = fit_preprocessor(_read_csv(args.input), _read_csv(args.history), config)
    bootstrap_summary: dict[str, int | float | bool] = {"applied": False}
    if args.bootstrap_minority:
        transformed, bootstrap_summary = bootstrap_minority_class(
            transformed,
            target_column=config.target_column,
            minority_to_majority_ratio=args.minority_to_majority_ratio,
            random_state=args.random_state,
        )
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
            "feature_schema_coverage": feature_schema_coverage(artifacts.feature_columns).to_dict(
                orient="records"
            ),
            "config": asdict(config),
            "minority_bootstrap": bootstrap_summary,
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


def _collinearity_command(args: argparse.Namespace) -> None:
    """Write VIF-first diagnostics for a transformed training feature file."""
    frame = _read_csv(args.input)
    metadata = {args.id_column, args.year_column, args.target_column, "pred_start_dt"}
    feature_columns = [column for column in frame.columns if column not in metadata]
    diagnostics = feature_group_collinearity(
        frame,
        feature_columns,
        max_rows=args.max_rows,
        random_state=args.random_state,
    )
    output_dir = project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "input": str(project_path(args.input)),
        "max_rows": args.max_rows,
        "groups": {},
    }
    for group, diagnostic in diagnostics.items():
        vif_path = output_dir / f"{group}_vif.csv"
        covariance_path = output_dir / f"{group}_covariance.csv"
        correlation_path = output_dir / f"{group}_correlation.csv"
        diagnostic["vif"].to_csv(vif_path, index=False)
        diagnostic["covariance"].to_csv(covariance_path)
        diagnostic["correlation"].to_csv(correlation_path)
        summary["groups"][group] = {
            "features": diagnostic["vif"]["feature"].tolist() if not diagnostic["vif"].empty else [],
            "vif": str(vif_path),
            "covariance": str(covariance_path),
            "correlation": str(correlation_path),
        }
    summary_path = write_json(args.summary_output, summary)
    print(f"Wrote collinearity diagnostics: {output_dir}")
    print(f"Wrote collinearity summary: {summary_path}")


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", required=True, help="Anchor CSV: train.csv, validation.csv, or test.csv.")
    parser.add_argument(
        "--history", required=True, help="Round-specific history.csv from make_walk_forward_splits."
    )
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
    fit.add_argument(
        "--feature-mode",
        choices=["decayed_history", "firm_year"],
        default="decayed_history",
        help=(
            "decayed_history averages up to four past/current statements; firm_year retains "
            "only direct anchor-year features while keeping backward-looking change features."
        ),
    )
    fit.add_argument("--target-encoding-smoothing", type=float, default=20.0)
    fit.add_argument("--target-encoding-folds", type=int, default=5)
    fit.add_argument("--random-state", type=int, default=42)
    fit.add_argument(
        "--bootstrap-minority",
        action="store_true",
        help="Optionally bootstrap only the training minority class after feature transformation.",
    )
    fit.add_argument(
        "--minority-to-majority-ratio",
        type=float,
        default=0.1,
        help="Target positive/minority rows per majority row when bootstrapping (0.1 = 1:10; 0.2 = 1:5).",
    )
    transform = commands.add_parser("transform", help="Transform validation/test data with fitted artifacts.")
    _add_common_arguments(transform)
    transform.add_argument("--artifacts", required=True, help="Joblib artifact input path.")
    collinearity = commands.add_parser(
        "collinearity",
        help="Write VIF, covariance, and correlation diagnostics for transformed training features.",
    )
    collinearity.add_argument("--input", required=True, help="Transformed training feature CSV.")
    collinearity.add_argument("--output-dir", required=True, help="Directory for grouped VIF/covariance CSVs.")
    collinearity.add_argument(
        "--summary-output",
        default="reports/metrics/feature_collinearity.json",
        help="JSON diagnostic manifest path.",
    )
    collinearity.add_argument("--id-column", default="id")
    collinearity.add_argument("--year-column", default="fs_year")
    collinearity.add_argument("--target-column", default="label")
    collinearity.add_argument("--max-rows", type=int, default=100_000)
    collinearity.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "fit":
        _fit_command(args)
    elif args.command == "transform":
        _transform_command(args)
    else:
        _collinearity_command(args)


if __name__ == "__main__":
    main()
