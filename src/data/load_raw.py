"""Robust reader for Banca Massiccia statement files (training CSV and, later, the holdout).

Written by data-auditor (2026-10-30). Unified NA handling, NO imputation.

- Reads every cell as a string first, then maps all missing-value encodings to NaN.
- Accepts either spelling of the three names that differ between the file and Appendix B
  (D-010) and an optional unnamed index column.
- Categoricals (HQ_city, legal_struct, ateco_sector) are kept as strings, never numbers.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

# Every token treated as missing (case-insensitive, after stripping whitespace).
# The training file only uses the empty string; the holdout "may use different ones"
# (Appendix B), so the list is deliberately broad.
NA_TOKENS = {
    "", "na", "n/a", "n.a.", "nan", "null", "none", "nil", "missing", "#n/a", "#na", "#value!",
    "#div/0!", "#ref!", "#num!", "-", "--", ".", "?", "inf", "-inf", "+inf", "infinity",
    "-infinity", "nd", "n.d.", "<na>", "nat",
}
# Candidate numeric sentinels: COUNTED, NOT masked. In training they occur only as plausible
# small euro amounts (e.g. -999 EUR of extraordinary income), so masking would destroy data.
NUMERIC_SENTINELS = [-999, -9999, -99999, -999999, 9999999999, -9999999999]

# File name  <- alternative spellings (Appendix B dictionary / typos)
ALIASES = {
    "eqty_tot": ["eqyt_tot"],
    "rev_operating": ["rev_operations"],
    "prof_operations": ["prof_operating"],
}

ID_COLS = ["id", "fs_year"]
DATE_COLS = ["stmt_date", "def_date"]
CAT_COLS = ["HQ_city", "legal_struct", "ateco_sector"]
NUM_COLS = [
    "asst_intang_fixed", "asst_tang_fixed", "asst_fixed_fin", "asst_current", "AR",
    "cash_and_equiv", "asst_tot", "eqty_tot", "eqty_corp_family_tot", "liab_lt", "liab_lt_emp",
    "debt_bank_st", "debt_bank_lt", "debt_fin_st", "debt_fin_lt", "AP_st", "AP_lt", "debt_st",
    "debt_lt", "rev_operating", "COGS", "prof_operations", "goodwill", "inc_financing",
    "exp_financing", "prof_financing", "inc_extraord", "taxes", "profit", "days_rec", "ebitda",
    "roa", "roe", "wc_net", "margin_fin", "cf_operations",
]


def _is_na_token(s: pd.Series) -> pd.Series:
    return s.str.strip().str.lower().isin(NA_TOKENS)


def _clean_code(s: pd.Series) -> pd.Series:
    """'28.0' -> '28', ' srl ' -> 'SRL'; NA tokens -> NaN. Keeps codes as strings."""
    s = s.astype("string").str.strip()
    s = s.mask(_is_na_token(s.fillna("")))
    s = s.str.replace(r"\.0+$", "", regex=True).str.upper()
    return s


def read_raw(path: str, nrows: int | None = None) -> tuple[pd.DataFrame, dict]:
    """Return (typed dataframe, report of NA encodings found)."""
    raw = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[], nrows=nrows)
    report: dict = {"n_rows": len(raw), "columns_in_file": list(raw.columns), "na_tokens": {}}

    # drop unnamed index column(s), keep as provenance
    idx_cols = [c for c in raw.columns if re.match(r"^Unnamed", c) or c == ""]
    out = pd.DataFrame(index=raw.index)
    if idx_cols:
        out["src_index"] = pd.to_numeric(raw[idx_cols[0]], errors="coerce").astype("Int64")
    report["index_cols_dropped"] = idx_cols

    # alias handling
    for canon, alts in ALIASES.items():
        if canon not in raw.columns:
            for a in alts:
                if a in raw.columns:
                    raw = raw.rename(columns={a: canon})
                    report.setdefault("renamed", {})[a] = canon

    for c in raw.columns:
        if c in idx_cols:
            continue
        s = raw[c]
        tok = s[_is_na_token(s)].str.strip().value_counts()
        if len(tok):
            report["na_tokens"][c] = {repr(k): int(v) for k, v in tok.items()}

    out["id"] = pd.to_numeric(raw["id"], errors="coerce").astype("Int64")
    out["fs_year"] = pd.to_numeric(raw["fs_year"], errors="coerce").astype("Int64")
    for c in DATE_COLS:
        if c in raw.columns:
            s = raw[c].mask(_is_na_token(raw[c]))
            out[c] = pd.to_datetime(s, errors="coerce")
        else:
            out[c] = pd.NaT
    for c in CAT_COLS:
        out[c] = _clean_code(raw[c]) if c in raw.columns else pd.Series(pd.NA, index=raw.index, dtype="string")
    sentinel_hits = {}
    for c in NUM_COLS:
        if c not in raw.columns:
            out[c] = np.nan
            continue
        s = raw[c].mask(_is_na_token(raw[c]))
        v = pd.to_numeric(s.str.replace(",", "", regex=False), errors="coerce").astype("float64")
        unparsable = int((v.isna() & s.notna()).sum())
        if unparsable:
            report.setdefault("unparsable_numeric", {})[c] = unparsable
        v = v.where(np.isfinite(v))
        hits = int(v.isin(NUMERIC_SENTINELS).sum())
        if hits:
            sentinel_hits[c] = hits
        out[c] = v
    report["numeric_sentinels_found"] = sentinel_hits
    report["extra_columns"] = [c for c in raw.columns if c not in idx_cols + ID_COLS + DATE_COLS + CAT_COLS + NUM_COLS]
    report["missing_expected_columns"] = [c for c in ID_COLS + DATE_COLS + CAT_COLS + NUM_COLS if c not in raw.columns]
    return out, report
