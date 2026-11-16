# Frozen notebook export; no fitting is performed by the harness.
import numpy as np
import pandas as pd
from bundle_paths import project_path

# Missing-value handling; see docs/project-update/features/02-missing-value.md
DROP_COLS = ["Unnamed: 0", "eqty_corp_family_tot", "days_rec"]
NON_NUMERIC = ["stmt_date", "legal_struct", "def_date", "pred_start_dt"]
DEBT_DETAIL = ["debt_bank_st", "debt_bank_lt", "debt_fin_st", "debt_fin_lt", "AP_st", "AP_lt"]
# Method F: fixed-asset columns, restored from eqty_tot - margin_fin, else 0
FIXED = ["asst_intang_fixed", "asst_tang_fixed", "asst_fixed_fin"]
# Method G: a missing value is 0 if the firm's previous-year value was 0 (past only, no look-ahead)
PREV_ZERO = ["liab_lt", "liab_lt_emp", "debt_lt"]
# Method H: debt-detail columns are 0 when their total is 0
DETAIL_OF = {"debt_st": ["debt_bank_st", "debt_fin_st", "AP_st"], "debt_lt": ["debt_bank_lt", "debt_fin_lt", "AP_lt"]}
# Method B: columns filled with (sector median of column / asst_tot) * asst_tot
B_COLS = [
    "asst_current", "AR", "cash_and_equiv", "goodwill",
    "eqty_tot", "liab_lt", "liab_lt_emp", *DEBT_DETAIL, "debt_st", "debt_lt",
    "rev_operating", "COGS", "prof_operations", "inc_financing", "exp_financing", "prof_financing",
    "inc_extraord", "taxes", "profit", "ebitda", "cf_operations",
]


def _ateco_industry_lookup():
    """Build detailed ATECO code to coarse A-U section mapping."""
    lookup = pd.read_csv(project_path("data/interim/ateco_sector_english.csv"), dtype={"code": "string"})
    codes = lookup["code"].str.strip()
    section = codes.where(codes.str.fullmatch(r"[A-U]")).ffill()
    rows = pd.DataFrame({"code": codes, "ateco_industry_group": section})
    return rows.loc[rows["code"].str.fullmatch(r"\d{2}")].set_index("code")["ateco_industry_group"]


def _ateco_industry_group(sector, lookup):
    numeric = pd.to_numeric(sector, errors="coerce")
    codes = numeric.where(numeric.mod(1).eq(0)).astype("Int64").astype("string").str.zfill(2)
    return codes.map(lookup)


def sector_medians(train):
    """Median table for method B, indexed by coarse A-U group; last row "ALL" is overall median.
    The asst_tot column holds the median amount (safeguard for holdout rows missing asst_tot); the other columns hold median ratios to asst_tot."""
    train = _to_numeric(train)
    ratio = train[B_COLS].div(train["asst_tot"], axis=0)
    ratio["asst_tot"] = train["asst_tot"]
    industry_group = _ateco_industry_group(train["ateco_sector"], _ateco_industry_lookup())
    med = ratio.groupby(industry_group).median()
    # roe is a percentage, not an amount: plain group median, no asst_tot scaling
    med["roe"] = train["roe"].groupby(industry_group).median()
    med.loc["ALL"] = ratio.median()
    med.loc["ALL", "roe"] = train["roe"].median()
    return med


def _to_numeric(df):
    # Blanks, "NA" and other missing-value spellings all become NaN
    df = df.copy()
    for c in df.columns.difference(NON_NUMERIC):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def fill_missing(df, medians):
    """Take raw data, return filled data. medians must be the result of sector_medians(training data)."""
    df = _to_numeric(df).drop(columns=DROP_COLS, errors="ignore")

    # Method A: a missing HQ_city becomes its own category
    df["HQ_city"] = df["HQ_city"].map("{:.0f}".format, na_action="ignore").fillna("Unknown")

    # Method C: record the flag before filling
    df["debt_detail_missing"] = df[DEBT_DETAIL].isna().any(axis=1).astype(int)

    # Method D: prof_operations = rev_operating - COGS; restore whichever of the three is missing
    df["rev_operating"] = df["rev_operating"].fillna(df["prof_operations"] + df["COGS"])
    df["COGS"] = df["COGS"].fillna(df["rev_operating"] - df["prof_operations"])
    df["prof_operations"] = df["prof_operations"].fillna(df["rev_operating"] - df["COGS"])

    # Method F: if exactly one fixed-asset column is missing, it is eqty_tot - margin_fin minus the other two; otherwise 0
    miss = df[FIXED].isna()
    est = (df["eqty_tot"] - df["margin_fin"] - df[FIXED].sum(axis=1)).clip(lower=0)
    for c in FIXED:
        df[c] = df[c].fillna(est.where(miss.sum(axis=1) == 1)).fillna(0)

    # Method G: previous year's value is 0 -> 0
    prev = df.sort_values(["id", "fs_year"]).groupby("id")[PREV_ZERO].shift(1).reindex(df.index)
    for c in PREV_ZERO:
        df[c] = df[c].fillna(prev[c].where(prev[c] == 0))

    # Method H: total debt is 0 -> its detail columns are 0
    for total, cols in DETAIL_OF.items():
        df[cols] = df[cols].mask(df[cols].isna() & (df[total] == 0).to_numpy()[:, None], 0)

    # Method B: look up the sector medians; sectors absent from training use the overall median
    industry_group = _ateco_industry_group(df["ateco_sector"], _ateco_industry_lookup())
    m = medians.drop(index="ALL").reindex(industry_group).set_axis(df.index).fillna(medians.loc["ALL"])
    df["asst_tot"] = df["asst_tot"].fillna(m["asst_tot"])
    df[B_COLS] = df[B_COLS].fillna(m[B_COLS].mul(df["asst_tot"], axis=0))
    # roe: plain industry-group median (a percentage, no asst_tot scaling)
    df["roe"] = df["roe"].fillna(m["roe"])

    # Method E: restore by formula (after B and D, so every input is already filled)
    fixed = df["asst_intang_fixed"] + df["asst_tang_fixed"] + df["asst_fixed_fin"]
    df["margin_fin"] = df["margin_fin"].fillna(df["eqty_tot"] - fixed)
    df["wc_net"] = df["wc_net"].fillna(df["asst_current"] - df["debt_st"])
    df["roa"] = df["roa"].fillna(df["prof_operations"] / df["asst_tot"] * 100)

    return df

