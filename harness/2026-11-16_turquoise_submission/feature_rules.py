"""Frozen feature rules from preprocess.py; inference uses no labels."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from bundle_paths import project_path
FEATURE_SCHEMA = json.loads(project_path("config/feature_schema.json").read_text())

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

def _region(city: pd.Series) -> pd.Series:
    """Map province codes to regions; missing or unrecognised codes become 'Unknown'."""
    return city.astype(str).map(REGION_BY_PROVINCE).fillna("Unknown")

def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide financial values, adding 1 to zero denominators to prevent errors.

    A zero denominator becomes 1, so the ratio degrades to the numerator rather
    than infinity; the corresponding ``zero_*`` flag still records the original
    zero. Reported negative denominators are divided as-is (the reported values
    are preserved, per the anomaly policy), and missing values stay NaN.
    """
    return numerator.div(denominator.mask(denominator == 0, denominator + 1))

def make_period_features(frame: pd.DataFrame, config: PreprocessingConfig) -> pd.DataFrame:
    """Create financial ratios and flags for each firm-year before history aggregation."""
    result = frame.sort_values([config.id_column, config.year_column]).copy()
    total_debt = result["debt_st"] + result["debt_lt"]
    bank_debt = result["debt_bank_st"] + result["debt_bank_lt"]
    long_term_capital = result["debt_lt"] + result["eqty_tot"]

    result["size"] = np.log(result["asst_tot"].where(result["asst_tot"] > 0))
    result["equity_ratio"] = _safe_divide(result["eqty_tot"], result["asst_tot"])
    result["debt_to_eqty"] = _safe_divide(total_debt, result["eqty_tot"])
    result["st_debt_share"] = _safe_divide(result["debt_st"], total_debt)
    result["lt_debt_to_capital"] = _safe_divide(result["debt_lt"], long_term_capital)
    result["bank_debt_conc"] = _safe_divide(bank_debt, total_debt)
    result["tangible_asset_ratio"] = _safe_divide(result["asst_tang_fixed"], result["asst_tot"])
    result["goodwill_ratio"] = _safe_divide(result["goodwill"], result["asst_tot"])
    result["cash_ratio"] = _safe_divide(result["cash_and_equiv"], result["asst_tot"])
    result["cash_to_financing_exp_ratio"] = _safe_divide(
        result["cash_and_equiv"], result["exp_financing"]
    )
    result["cur_asst_ratio"] = _safe_divide(result["asst_current"], result["asst_tot"])
    result["AR_ratio"] = _safe_divide(result["AR"], result["rev_operating"])
    result["payables_ratio"] = _safe_divide(result["AP_st"], result["asst_tot"])
    result["wc_ratio"] = _safe_divide(result["wc_net"], result["asst_tot"])
    result["cur_debt_ratio"] = _safe_divide(result["asst_current"], result["debt_st"])
    result["quick_ratio"] = _safe_divide(result["cash_and_equiv"] + result["AR"], result["debt_st"])
    result["asset_turnover"] = _safe_divide(result["rev_operating"], result["asst_tot"])
    result["op_margin"] = _safe_divide(result["prof_operations"], result["rev_operating"])
    result["net_prof_margin"] = _safe_divide(result["profit"], result["rev_operating"])
    result["int_coverage"] = _safe_divide(result["ebitda"], result["exp_financing"])
    result["financing_expense_ratio"] = _safe_divide(result["exp_financing"], result["asst_tot"])
    # Inverted Debt/EBITDA: firms with EBITDA <= 0 keep a graded value instead of NaN.
    result["ebitda_to_debt"] = _safe_divide(result["ebitda"], total_debt)
    result["operating_cash_flow_margin"] = _safe_divide(result["cf_operations"], result["rev_operating"])
    result["cash_interest_coverage"] = _safe_divide(result["cf_operations"], result["exp_financing"])
    result["operating_cash_flow_ratio"] = _safe_divide(result["cf_operations"], result["asst_tot"])
    result["operating_cash_flow_to_debt"] = _safe_divide(result["cf_operations"], total_debt)
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

def _validate_period_feature_contract(frame: pd.DataFrame) -> None:
    """Fail clearly when a configured engineered feature was renamed or omitted."""
    expected = set(_period_numeric_features()) | set(_period_binary_features())
    missing = sorted(expected.difference(frame.columns))
    if missing:
        raise ValueError(
            "Feature schema does not match make_period_features output; "
            f"missing configured features: {missing}"
        )
