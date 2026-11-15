"""Fit and apply leakage-safe cleaning and feature engineering.

Each input row is a prediction anchor identified by ``(id, fs_year)``. The
output has one row per anchor. ``decayed_history`` aggregates financial
features from up to four statements whose fiscal year is no later than the
anchor. ``firm_year`` retains only the anchor-year features. The weight
for a statement ``k`` calendar years old is ``phi ** k``; available weights
are re-normalized for every feature in ``decayed_history`` mode.

Use ``fit`` on a round's train.csv only. It learns industry-group-median
imputation, industry target encoding, and legal-structure categories, writes
those artifacts, and transforms train.csv. Use ``transform`` with that artifact for
the corresponding validation.csv and test.csv. Both commands require the
round's history.csv, but labels in history are never used as input features.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

from finance_project.common import project_path, write_json


FEATURE_SCHEMA_PATH = project_path("config/feature_schema.json")
ATECO_INDUSTRY_LOOKUP_PATH = project_path("data/interim/ateco_sector_english.csv")


def _load_feature_schema(path: Path = FEATURE_SCHEMA_PATH) -> tuple[dict[str, Any], str]:
    """Load and validate the version-controlled feature contract."""
    if not path.exists():
        raise FileNotFoundError(f"Feature schema configuration not found: {path}")
    schema = json.loads(path.read_text(encoding="utf-8"))
    required_keys = {
        "schema_version",
        "debt_detail_columns",
        "anomaly_flags",
        "impute_columns",
        "raw_required",
        "derived_categorical_features",
        "period_numeric_features",
        "period_binary_features",
        "feature_groups",
        "mode_features",
        "dynamic_feature_prefixes",
    }
    missing_keys = sorted(required_keys.difference(schema))
    if missing_keys:
        raise ValueError(f"Feature schema is missing required keys: {missing_keys}")
    list_keys = required_keys.difference({"schema_version", "feature_groups", "mode_features"})
    for key in list_keys:
        values = schema[key]
        if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
            raise TypeError(f"Feature schema '{key}' must be a list of strings.")
        if len(values) != len(set(values)):
            raise ValueError(f"Feature schema '{key}' contains duplicate names.")
    if not isinstance(schema["schema_version"], str) or not schema["schema_version"]:
        raise TypeError("Feature schema 'schema_version' must be a non-empty string.")
    if not isinstance(schema["feature_groups"], dict) or not schema["feature_groups"]:
        raise TypeError("Feature schema 'feature_groups' must be a non-empty object.")
    if not isinstance(schema["mode_features"], dict):
        raise TypeError("Feature schema 'mode_features' must be an object.")
    for group, values in schema["feature_groups"].items():
        if not isinstance(group, str) or not isinstance(values, list) or not all(
            isinstance(value, str) for value in values
        ):
            raise TypeError("Every feature-group name and requirement must be a string.")
    for mode in {"decayed_history", "firm_year"}:
        if mode not in schema["mode_features"]:
            raise ValueError(f"Feature schema 'mode_features' is missing '{mode}'.")
        values = schema["mode_features"][mode]
        if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
            raise TypeError(f"Feature schema mode '{mode}' must be a list of strings.")
    if not set(schema["anomaly_flags"]).issubset(schema["period_binary_features"]):
        raise ValueError("Every configured anomaly flag must be a configured binary feature.")
    canonical_schema = json.dumps(schema, sort_keys=True, separators=(",", ":"))
    return schema, hashlib.sha256(canonical_schema.encode("utf-8")).hexdigest()


FEATURE_SCHEMA, FEATURE_SCHEMA_HASH = _load_feature_schema()
FEATURE_SCHEMA_VERSION = FEATURE_SCHEMA["schema_version"]
DEBT_DETAIL = list(FEATURE_SCHEMA["debt_detail_columns"])
ANOMALY_FLAGS = list(FEATURE_SCHEMA["anomaly_flags"])
IMPUTE_COLS = list(FEATURE_SCHEMA["impute_columns"])
REQUIRED_COLUMNS = set(FEATURE_SCHEMA["raw_required"]) | set(IMPUTE_COLS)
DERIVED_CATEGORICAL_FEATURES = list(FEATURE_SCHEMA["derived_categorical_features"])
ATECO_INDUSTRY_COLUMN = "ateco_industry_group"
if ATECO_INDUSTRY_COLUMN not in DERIVED_CATEGORICAL_FEATURES:
    raise ValueError(
        f"Feature schema must configure the derived categorical feature '{ATECO_INDUSTRY_COLUMN}'."
    )
FEATURE_GROUPS: dict[str, list[str]] = {
    group: list(requirements) for group, requirements in FEATURE_SCHEMA["feature_groups"].items()
}

# ISTAT province code (HQ_city, Allegati_in_italiano.pdf, Allegato 2) -> region.
_REGION_PROVINCES = {
    "Piemonte": [1, 2, 3, 4, 5, 6, 96, 103],
    "Valle d'Aosta": [7],
    "Lombardia": [*range(12, 21), 97, 98, 108],
    "Trentino-Alto Adige": [21, 22],
    "Veneto": list(range(23, 30)),
    "Friuli-Venezia Giulia": [30, 31, 32, 93],
    "Liguria": [8, 9, 10, 11],
    "Emilia-Romagna": [*range(33, 41), 99],
    "Toscana": [*range(45, 54), 100],
    "Umbria": [54, 55],
    "Marche": [41, 42, 43, 44, 109],
    "Lazio": [56, 57, 58, 59, 60],
    "Abruzzo": [66, 67, 68, 69],
    "Molise": [70, 94],
    "Campania": [61, 62, 63, 64, 65],
    "Puglia": [71, 72, 73, 74, 75, 110],
    "Basilicata": [76, 77],
    "Calabria": [78, 79, 80, 101, 102],
    "Sicilia": list(range(81, 90)),
    "Sardegna": [90, 91, 92, 95, 104, 105, 106, 107],
}
REGION_BY_PROVINCE = {
    str(code): region for region, codes in _REGION_PROVINCES.items() for code in codes
}
assert sorted(map(int, REGION_BY_PROVINCE)) == list(range(1, 111))


def _load_ateco_industry_mapping(path: Path = ATECO_INDUSTRY_LOOKUP_PATH) -> pd.Series:
    """Map detailed two-digit ATECO codes to their 21 A--U industry sections."""
    if not path.exists():
        raise FileNotFoundError(f"ATECO industry lookup not found: {path}")
    lookup = pd.read_csv(path, dtype={"code": "string"})
    if "code" not in lookup:
        raise ValueError("ATECO industry lookup must contain a 'code' column.")
    codes = lookup["code"].str.strip()
    sections = codes.where(codes.str.fullmatch(r"[A-U]")).ffill()
    mapping = (
        pd.DataFrame({"code": codes, ATECO_INDUSTRY_COLUMN: sections})
        .loc[lambda frame: frame["code"].str.fullmatch(r"\d{2}")]
        .set_index("code")[ATECO_INDUSTRY_COLUMN]
    )
    if mapping.isna().any() or sorted(mapping.unique()) != list("ABCDEFGHIJKLMNOPQRSTU"):
        raise ValueError("ATECO industry lookup must map detailed codes to all 21 A--U sections.")
    return mapping


ATECO_TO_INDUSTRY = _load_ateco_industry_mapping()


def _ateco_industry_group(sector: pd.Series) -> pd.Series:
    """Convert raw numeric ATECO codes, such as 14.0, to A--U industry sections."""
    numeric = pd.to_numeric(sector, errors="coerce")
    two_digit_codes = (
        numeric.where(numeric.mod(1).eq(0)).astype("Int64").astype("string").str.zfill(2)
    )
    return two_digit_codes.map(ATECO_TO_INDUSTRY)


def _region(city: pd.Series) -> pd.Series:
    """Map province codes to regions; missing or unrecognised codes become 'Unknown'."""
    return city.astype(str).map(REGION_BY_PROVINCE).fillna("Unknown")


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
        if self.feature_mode not in FEATURE_SCHEMA["mode_features"]:
            modes = sorted(FEATURE_SCHEMA["mode_features"])
            raise ValueError(f"feature_mode must be one of {modes}.")


@dataclass
class PreprocessingArtifacts:
    """Parameters learned exclusively from one round's training anchors."""

    config: PreprocessingConfig
    feature_schema_version: str
    feature_schema_hash: str
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
    namespace: dict[str, Any] = {"pd": pd, "np": np, "project_path": project_path}
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
    result[ATECO_INDUSTRY_COLUMN] = _ateco_industry_group(result[config.sector_column])
    return result


def _positive_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide financial values, adding 1 to zero denominators to prevent errors.

    A zero denominator becomes 1, so the ratio degrades to the numerator rather
    than NaN or infinity; the corresponding ``zero_*`` flag still records the
    original zero. Negative denominators remain economically invalid and yield
    NaN, as do missing values.
    """
    adjusted = denominator.mask(denominator == 0, denominator + 1)
    return numerator.div(adjusted.where(adjusted > 0))


def make_period_features(frame: pd.DataFrame, config: PreprocessingConfig) -> pd.DataFrame:
    """Create financial ratios and flags for each firm-year before history aggregation."""
    result = frame.sort_values([config.id_column, config.year_column]).copy()
    total_debt = result["debt_st"] + result["debt_lt"]
    bank_debt = result["debt_bank_st"] + result["debt_bank_lt"]
    long_term_capital = result["debt_lt"] + result["eqty_tot"]

    result["size"] = np.log(result["asst_tot"].where(result["asst_tot"] > 0))
    result["equity_ratio"] = _positive_divide(result["eqty_tot"], result["asst_tot"])
    result["debt_to_eqty"] = _positive_divide(total_debt, result["eqty_tot"])
    result["st_debt_share"] = _positive_divide(result["debt_st"], total_debt)
    result["lt_debt_to_capital"] = _positive_divide(result["debt_lt"], long_term_capital)
    result["bank_debt_conc"] = _positive_divide(bank_debt, total_debt)
    result["tangible_asset_ratio"] = _positive_divide(result["asst_tang_fixed"], result["asst_tot"])
    result["goodwill_ratio"] = _positive_divide(result["goodwill"], result["asst_tot"])
    result["cash_ratio"] = _positive_divide(result["cash_and_equiv"], result["asst_tot"])
    result["cash_to_financing_exp_ratio"] = _positive_divide(
        result["cash_and_equiv"], result["exp_financing"]
    )
    result["cur_asst_ratio"] = _positive_divide(result["asst_current"], result["asst_tot"])
    result["AR_ratio"] = _positive_divide(result["AR"], result["rev_operating"])
    result["payables_ratio"] = _positive_divide(result["AP_st"], result["asst_tot"])
    result["wc_ratio"] = _positive_divide(result["wc_net"], result["asst_tot"])
    result["cur_debt_ratio"] = _positive_divide(result["asst_current"], result["debt_st"])
    result["quick_ratio"] = _positive_divide(result["cash_and_equiv"] + result["AR"], result["debt_st"])
    result["asset_turnover"] = _positive_divide(result["rev_operating"], result["asst_tot"])
    result["op_margin"] = _positive_divide(result["prof_operations"], result["rev_operating"])
    result["net_prof_margin"] = _positive_divide(result["profit"], result["rev_operating"])
    result["int_coverage"] = _positive_divide(result["ebitda"], result["exp_financing"])
    result["financing_expense_ratio"] = _positive_divide(result["exp_financing"], result["asst_tot"])
    # Inverted Debt/EBITDA: firms with EBITDA <= 0 keep a graded value instead of NaN.
    result["ebitda_to_debt"] = _positive_divide(result["ebitda"], total_debt)
    result["operating_cash_flow_margin"] = _positive_divide(result["cf_operations"], result["rev_operating"])
    result["cash_interest_coverage"] = _positive_divide(result["cf_operations"], result["exp_financing"])
    result["operating_cash_flow_ratio"] = _positive_divide(result["cf_operations"], result["asst_tot"])
    result["operating_cash_flow_to_debt"] = _positive_divide(result["cf_operations"], total_debt)
    result["loss_flag"] = (result["profit"] < 0).astype(int)
    # Negative P&L flag per the feature schema: reported negative revenue, COGS,
    # or financing expense. Imputation never alters reported values, so this
    # reflects the original statement.
    result["neg_pl_flag"] = (
        (result["rev_operating"] < 0) | (result["COGS"] < 0) | (result["exp_financing"] < 0)
    ).astype(int)

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
    _validate_period_feature_contract(result)
    return result


def _period_numeric_features() -> list[str]:
    return list(FEATURE_SCHEMA["period_numeric_features"])


def _period_binary_features() -> list[str]:
    return list(FEATURE_SCHEMA["period_binary_features"])


def feature_schema_metadata() -> dict[str, str]:
    """Return the schema identity stored with preprocessing/model artifacts."""
    return {
        "path": str(FEATURE_SCHEMA_PATH),
        "version": FEATURE_SCHEMA_VERSION,
        "sha256": FEATURE_SCHEMA_HASH,
    }


def _validate_period_feature_contract(frame: pd.DataFrame) -> None:
    """Fail clearly when a configured engineered feature was renamed or omitted."""
    expected = set(_period_numeric_features()) | set(_period_binary_features())
    missing = sorted(expected.difference(frame.columns))
    if missing:
        raise ValueError(
            "Feature schema does not match make_period_features output; "
            f"missing configured features: {missing}"
        )


def _assert_artifact_schema_compatible(artifacts: PreprocessingArtifacts) -> None:
    """Prevent scoring with an artifact fitted under a different feature contract."""
    artifact_version = getattr(artifacts, "feature_schema_version", None)
    artifact_hash = getattr(artifacts, "feature_schema_hash", None)
    if artifact_version != FEATURE_SCHEMA_VERSION or artifact_hash != FEATURE_SCHEMA_HASH:
        raise ValueError(
            "Preprocessing artifact feature schema does not match the current configuration. "
            "Refit the preprocessor/model or restore the feature_schema.json used for fitting."
        )


def feature_schema_coverage(feature_columns: list[str]) -> pd.DataFrame:
    """Audit implemented model features against the configured feature contract.

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
        ATECO_INDUSTRY_COLUMN,
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
    """Keep direct anchor-year features.

    ``make_period_features`` is evaluated on the supplied history; this
    function then selects the exact ``(id, fs_year)`` anchor row rather than
    decay-averaging earlier values.
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
        ATECO_INDUSTRY_COLUMN,
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
    category = frame[ATECO_INDUSTRY_COLUMN]
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
        stats = fit.groupby(ATECO_INDUSTRY_COLUMN)[config.target_column].agg(["sum", "count"])
        mapping = (stats["sum"] + config.target_encoding_smoothing * fold_mean) / (
            stats["count"] + config.target_encoding_smoothing
        )
        encoded.iloc[validation_index] = category.iloc[validation_index].map(mapping).fillna(fold_mean)

    full_stats = frame.groupby(ATECO_INDUSTRY_COLUMN)[config.target_column].agg(["sum", "count"])
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
        output["ateco_sector_te"] = output[ATECO_INDUSTRY_COLUMN].map(
            artifacts.industry_mapping
        ).fillna(artifacts.industry_global_mean)

    legal = output[config.legal_column].fillna("Unknown").astype(str)
    for category in artifacts.legal_categories:
        output[f"legal_{category}"] = legal.eq(category).astype(int)
    city = _region(output["HQ_city"])
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


def _report_feature_quality(
    frame: pd.DataFrame,
    feature_columns: list[str],
    *,
    stage: str,
) -> None:
    """Print missing and infinite value counts for transformed feature columns.

    Reported only; nothing is dropped or filled here. Missing ratios (for
    example, from negative denominators or first-observation change features)
    are legitimate model inputs for tree-based models.
    """
    features = frame.reindex(columns=feature_columns).apply(pd.to_numeric, errors="coerce")
    rows = len(features)
    missing = features.isna().sum()
    missing = missing.loc[missing > 0].sort_values(ascending=False)
    infinite = pd.Series(
        {column: int(np.isinf(features[column]).sum()) for column in features.columns}
    )
    infinite = infinite.loc[infinite > 0].sort_values(ascending=False)

    print(f"[{stage}] Post-transformation feature check: {rows} rows x {len(feature_columns)} features.")
    for label, counts in (("Missing", missing), ("Infinite", infinite)):
        if counts.empty:
            print(f"[{stage}] {label.lower()} feature values: none.")
            continue
        print(f"[{stage}] {label.lower()} feature values in {len(counts)} column(s):")
        for column, count in counts.items():
            share = count / rows if rows else 0.0
            print(f"  {column}: {int(count)} rows ({share:.2%})")


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
    all_city_categories = sorted(_region(engineered["HQ_city"]).unique())
    legal_reference_category, *legal_categories = all_legal_categories
    city_reference_category, *city_categories = all_city_categories
    feature_columns = [
        *_period_numeric_features(),
        *_period_binary_features(),
    ]
    feature_columns.extend(FEATURE_SCHEMA["mode_features"][config.feature_mode])
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
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        feature_schema_hash=FEATURE_SCHEMA_HASH,
        sector_medians=medians,
        industry_mapping=industry_mapping,
        industry_global_mean=industry_global_mean,
        legal_categories=legal_categories,
        legal_reference_category=legal_reference_category,
        city_categories=city_categories,
        city_reference_category=city_reference_category,
        feature_columns=feature_columns,
    )
    transformed = _finalize_features(engineered, artifacts, training=True)
    _report_feature_quality(transformed, artifacts.feature_columns, stage="fit")
    return transformed, artifacts


def transform_with_preprocessor(
    anchors: pd.DataFrame,
    history: pd.DataFrame,
    artifacts: PreprocessingArtifacts,
) -> pd.DataFrame:
    """Transform validation, test, or holdout anchors without refitting any statistics."""
    _assert_artifact_schema_compatible(artifacts)
    config = artifacts.config
    anchors = _drop_index_columns(anchors)
    history = _drop_index_columns(history)
    _require_columns(anchors, include_target=False, config=config)
    _require_columns(history, include_target=False, config=config)
    engineered = _build_features(anchors, history, artifacts.sector_medians, config)
    transformed = _finalize_features(engineered, artifacts, training=False)
    _report_feature_quality(transformed, artifacts.feature_columns, stage="transform")
    return transformed


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
            "feature_schema": feature_schema_metadata(),
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
