"""Feature layer for the Banca Massiccia 1-year PD model (feature-architect, 2026-10-30).

Design
------
    params = fit(train_df)                 # learns ONLY from the rows passed in (training window)
    X      = transform(df, params)         # pure function of (one row, params); never batch stats
    X_imp  = transform(df, params, impute=True)   # same, NaNs filled (for logit / probit)
    Z      = design_matrix(X_imp, params)  # one-hot categoricals, float matrix for GLMs

`params` is a plain JSON-serialisable dict (save_params / load_params).

What happens in transform, in order
-----------------------------------
1. `coerce_input`: accept either column spelling (D-010), ignore extra/index columns, turn every
   missing-value token ("NA", "", "n.a.", "#DIV/0!", ...) and every non-finite number into NaN,
   keep categorical codes as strings ("28.0" -> "28"). Missing columns become all-NaN.
2. `_reconstruct`: fill a missing raw field from an exact accounting identity when the other
   fields are present (Roger G2; identities validated in reports/data/01_profile.md section 5).
3. Ratios with explicit, documented rules for zero / negative denominators (RATIO RULES below).
4. Winsorise each continuous feature to [p1, p99] of the TRAINING window (bounds in params,
   on the raw ratio scale). +inf / -inf produced by a documented rule land on the bound.
   Ratios with unbounded denominators (coverage, current ratio, margin, days) are then put
   on a signed-log scale (`slog`, monotone).
5. Flags (negative equity, debt-split block missing, non-positive revenue) and categoricals
   (ATECO section group, legal-form group, macro-area; unseen -> "OTHER").
6. Optional `impute=True`: debt-split components missing as a block are rebuilt as
   (train-window sector median share of the total) x (the firm's own debt_st / debt_lt), then
   any remaining NaN takes the train-window sector median of that feature (global median
   fallback). The missing flag stays in X so the model can price the missingness itself.

Nothing in this module reads labels at transform time, nor any row other than the current one.
Growth / lagged features are deliberately absent (D-020): the current record has no prior-year
fields and the harness may not use a firm's other holdout rows.
"""
from __future__ import annotations

import json
import re
from typing import Any

import numpy as np
import pandas as pd

from src.data.codes import REGION_MACRO, ateco_section, province_region
from src.data.load_raw import ALIASES, NA_TOKENS

VERSION = "features-v1 (2026-10-30)"
WINSOR_Q = (0.01, 0.99)
MIN_DEFAULTS_PER_LEVEL = 30      # categorical levels with fewer train defaults are pooled
OTHER = "OTHER"

RAW_NUM = [
    "asst_intang_fixed", "asst_tang_fixed", "asst_fixed_fin", "asst_current", "AR",
    "cash_and_equiv", "asst_tot", "eqty_tot", "liab_lt", "liab_lt_emp", "debt_bank_st",
    "debt_bank_lt", "debt_fin_st", "debt_fin_lt", "AP_st", "AP_lt", "debt_st", "debt_lt",
    "rev_operating", "COGS", "prof_operations", "goodwill", "inc_financing", "exp_financing",
    "prof_financing", "inc_extraord", "taxes", "profit", "ebitda", "roa", "roe", "wc_net",
    "margin_fin", "cf_operations",
]
RAW_CAT = ["HQ_city", "legal_struct", "ateco_sector"]
ST_BLOCK = ["debt_bank_st", "debt_fin_st", "AP_st"]
LT_BLOCK = ["debt_bank_lt", "debt_fin_lt", "AP_lt"]

# ---------------------------------------------------------------------------------------------
# Feature metadata: the single source of truth for docs/FEATURE_SPEC.md, monotone constraints
# and the referee's face-validity check.
#   sign: +1 = PD rises with the feature, -1 = PD falls, 0 = no monotone expectation.
#   core: recommended for the probit / logit core set (GBM uses all).
# ---------------------------------------------------------------------------------------------
FEATURES: list[dict[str, Any]] = [
    # ---- size ---------------------------------------------------------------------------
    dict(name="log_assets", family="size", sign=-1, core=True,
         formula="ln(asst_tot)",
         reason="Larger firms are more diversified and have better access to funding; size "
                "enters the model once, here (all other features are scale-free ratios).",
         zero_rule="asst_tot <= 0 or missing -> rebuilt from the four asset components; else NaN"),
    # ---- leverage / capital structure ---------------------------------------------------
    dict(name="equity_ratio", family="leverage", sign=-1, core=True,
         formula="eqty_tot / asst_tot",
         reason="Book equity cushion: the share of assets that can be lost before creditors "
                "are hit. Bounded alternative to debt/equity, defined for negative equity.",
         zero_rule="denominator is total assets (> 0); negative equity kept as a negative ratio"),
    dict(name="st_debt_ratio", family="leverage", sign=+1, core=True,
         formula="debt_st / asst_tot",
         reason="Short-term liabilities (all payables due within 12 months) relative to assets: "
                "the claims that must be refinanced or paid inside the PD horizon.",
         zero_rule="denominator is total assets (> 0)"),
    dict(name="bank_debt_ratio", family="leverage", sign=+1, core=False,
         formula="(debt_bank_st + debt_bank_lt) / asst_tot",
         reason="Bank borrowing relative to assets: the lender-facing leverage that triggers "
                "a missed payment.",
         zero_rule="denominator is total assets; NaN if the debt-split block is missing "
                   "(impute mode: sector share x debt_st / debt_lt)"),
    dict(name="st_debt_share", family="leverage", sign=+1, core=False,
         formula="debt_st / (debt_st + debt_lt)",
         reason="Maturity structure: a larger short-term share means more refinancing within "
                "the year.",
         zero_rule="denominator <= 0 -> NaN (no debt at all, or a sign error)"),
    # ---- liquidity ----------------------------------------------------------------------
    dict(name="wc_ratio", family="liquidity", sign=-1, core=False,
         formula="wc_net / asst_tot  (= (asst_current - debt_st) / asst_tot)",
         reason="Net working capital: current assets left after paying short-term "
                "liabilities, scaled by assets; no zero-denominator problem.",
         zero_rule="denominator is total assets (> 0)"),
    dict(name="current_ratio", slog=True, family="liquidity", sign=-1, core=False,
         formula="slog(asst_current / debt_st)",
         reason="Classic liquidity ratio: current assets per euro of short-term liabilities.",
         zero_rule="debt_st == 0 -> +inf (no short-term claims) -> clipped to the upper bound; "
                   "debt_st < 0 -> NaN (sign error)"),
    dict(name="cash_ratio", family="liquidity", sign=-1, core=True,
         formula="cash_and_equiv / asst_tot",
         reason="Immediate liquidity available to meet the next payments.",
         zero_rule="denominator is total assets (> 0)"),
    # ---- profitability ------------------------------------------------------------------
    dict(name="roa_net", family="profitability", sign=-1, core=True,
         formula="profit / asst_tot",
         reason="Net return on assets after interest, extraordinary items and tax: losses "
                "erode equity and the capacity to pay.",
         zero_rule="denominator is total assets (> 0)"),
    dict(name="roa_oper", family="profitability", sign=-1, core=False,
         formula="prof_operations / asst_tot  (= roa / 100 in the file)",
         reason="Operating return on assets, before financing and extraordinary items.",
         zero_rule="denominator is total assets (> 0)"),
    dict(name="ebitda_margin", slog=True, family="profitability", sign=-1, core=False,
         formula="slog(ebitda / rev_operating)",
         reason="Operating cash margin per euro of sales.",
         zero_rule="rev_operating <= 0 -> NaN (margin undefined; see flag rev_nonpositive)"),
    # ---- debt coverage / debt service ---------------------------------------------------
    dict(name="interest_coverage", slog=True, family="coverage", sign=-1, core=True,
         formula="slog(ebitda / exp_financing)",
         reason="How many times operating cash earnings cover interest expense: the direct "
                "measure of the ability to service debt.",
         zero_rule="exp_financing <= 0 (no interest expense, or sign error): +inf if ebitda > 0 "
                   "(clipped to upper bound), 0 if ebitda <= 0; negative EBITDA with positive "
                   "interest gives a negative ratio (worst)."),
    dict(name="cfo_to_debt", family="coverage", sign=-1, core=False,
         formula="cf_operations / (debt_st + debt_lt)",
         reason="Accounting cash flow (net profit + D&A) per euro of debt: the years needed to "
                "repay debt from internal funds, inverted.",
         zero_rule="debt == 0: +inf if cf_operations > 0 else 0; debt < 0 -> NaN"),
    dict(name="interest_to_assets", family="coverage", sign=+1, core=True,
         formula="exp_financing / asst_tot",
         reason="Interest burden scaled by assets (= cost of debt x leverage). Converts the "
                "raw euro amount exp_financing into a ratio (Roger M9).",
         zero_rule="denominator is total assets (> 0)"),
    # ---- activity / efficiency ----------------------------------------------------------
    dict(name="asset_turnover", family="activity", sign=0, core=False,
         formula="rev_operating / asst_tot",
         reason="Sales per euro of assets. A priori '-' (idle or impaired assets), but high "
                "turnover also marks thin-margin trading firms; training data show an inverted U "
                "(D-022), so no monotone constraint.",
         zero_rule="denominator is total assets (> 0)"),
    dict(name="ar_days", slog=True, family="activity", sign=+1, core=True,
         formula="slog(365 * AR / rev_operating)  (replaces the file's days_rec)",
         reason="Days of sales outstanding: slow collection ties up cash and can signal weak "
                "customers or overstated receivables.",
         zero_rule="rev_operating == 0: +inf if AR > 0 else 0; rev_operating < 0 -> NaN"),
    dict(name="ap_days", slog=True, family="activity", sign=+1, core=False,
         formula="slog(365 * AP_st / COGS)",
         reason="Days payable outstanding: stretching suppliers is a typical symptom of "
                "liquidity stress.",
         zero_rule="COGS <= 0 -> NaN; NaN if the debt-split block is missing"),
    # ---- asset structure ----------------------------------------------------------------
    dict(name="tangible_ratio", family="asset structure", sign=-1, core=True,
         formula="asst_tang_fixed / asst_tot",
         reason="Tangible fixed assets are pledgeable collateral and retain value in distress.",
         zero_rule="denominator is total assets (> 0)"),
    dict(name="intangible_ratio", family="asset structure", sign=+1, core=False,
         formula="asst_intang_fixed / asst_tot",
         reason="Intangibles (capitalised costs, brands) lose most of their value in distress.",
         zero_rule="denominator is total assets (> 0)"),
    dict(name="extraord_to_assets", family="profitability", sign=0, core=False,
         formula="inc_extraord / asst_tot",
         reason="Extraordinary items are two-sided: large losses (write-downs) and large gains "
                "(asset sales under pressure) can both precede default. No monotone constraint.",
         zero_rule="denominator is total assets (> 0)"),
    # ---- flags --------------------------------------------------------------------------
    dict(name="neg_equity", family="leverage (flag)", sign=+1, core=True, kind="flag",
         formula="1 if eqty_tot < 0",
         reason="Negative book equity: liabilities exceed assets, so creditors are already "
                "under-covered in book terms; adds a jump that winsorised equity_ratio cannot show.",
         zero_rule="NaN equity -> 0 after reconstruction attempt"),
    dict(name="debt_split_missing", family="missing indicator", sign=+1, core=False, kind="flag",
         formula="1 if any of debt_bank_st, debt_bank_lt, AP_st is missing",
         reason="The bank/financial/trade debt split is missing as a block, more often for small "
                "firms and persistently within firm (D-016); weakly riskier. Lets the model price "
                "the missingness after imputation.",
         zero_rule="n/a"),
    dict(name="rev_nonpositive", family="activity (flag)", sign=0, core=False, kind="flag",
         formula="1 if rev_operating <= 0",
         reason="No (or negative) operating revenue: holding / start-up / wind-down firms. "
                "Marks where revenue-based ratios are undefined. Direction mixed (0 vs < 0).",
         zero_rule="n/a"),
    # ---- categoricals -------------------------------------------------------------------
    dict(name="ateco_group", family="industry", sign=0, core=True, kind="cat",
         formula="ATECO 2007 division -> section A-U; sections with < 30 train defaults -> OTHER",
         reason="Industry default rates differ (e.g. real estate vs manufacturing); RiskCalc "
                "adds industry indicators to the probit (R-01 Sec. 3.1, p. 14).",
         zero_rule="missing / unparsable / unseen section -> OTHER"),
    dict(name="legal_group", family="firm type", sign=0, core=True, kind="cat",
         formula="legal_struct; forms with < 30 train defaults -> OTHER",
         reason="Single-member S.r.l. vs S.p.A. differ in governance and capital requirements "
                "[code meanings UNVERIFIED].",
         zero_rule="missing / unseen -> OTHER"),
    dict(name="macro_area", family="geography", sign=0, core=True, kind="cat",
         formula="HQ_city (ISTAT province code) -> region -> NUTS-1 macro-area",
         reason="Regional economic conditions; 5 levels keep every level well populated.",
         zero_rule="missing / unknown province code -> OTHER"),
]
META = {f["name"]: {**{"kind": "num"}, **f} for f in FEATURES}
NUM_FEATURES = [f["name"] for f in FEATURES if f.get("kind", "num") == "num"]
FLAG_FEATURES = [f["name"] for f in FEATURES if f.get("kind") == "flag"]
CAT_FEATURES = [f["name"] for f in FEATURES if f.get("kind") == "cat"]
ALL_FEATURES = NUM_FEATURES + FLAG_FEATURES + CAT_FEATURES
CORE_FEATURES = [f["name"] for f in FEATURES if f["core"]]
BLOCK_FEATURES = ["bank_debt_ratio", "ap_days"]


# ---------------------------------------------------------------------------------------------
# Input cleaning
# ---------------------------------------------------------------------------------------------
def _to_num(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        v = pd.to_numeric(s, errors="coerce").astype("float64")
    else:
        t = s.astype("string").str.strip()
        t = t.mask(t.str.lower().isin(NA_TOKENS))
        t = t.str.replace(",", "", regex=False)
        v = pd.to_numeric(t, errors="coerce").astype("float64")
    return v.where(np.isfinite(v))


def _to_code(s: pd.Series) -> pd.Series:
    t = s.astype("string").str.strip()
    t = t.mask(t.fillna("").str.lower().isin(NA_TOKENS))
    return t.str.replace(r"\.0+$", "", regex=True).str.upper()


def coerce_input(df: pd.DataFrame) -> pd.DataFrame:
    """Return a clean frame with exactly RAW_NUM (float) + RAW_CAT (string) columns, same index."""
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    cols = {}
    lower = {str(c).strip().lower(): c for c in df.columns}
    for canon in RAW_NUM + RAW_CAT:
        src = None
        for cand in [canon] + ALIASES.get(canon, []):
            if cand in df.columns:
                src = cand
                break
            if cand.lower() in lower:
                src = lower[cand.lower()]
                break
        cols[canon] = src
    out = pd.DataFrame(index=df.index)
    for c in RAW_NUM:
        out[c] = _to_num(df[cols[c]]) if cols[c] is not None else np.nan
    for c in RAW_CAT:
        out[c] = (_to_code(df[cols[c]]) if cols[c] is not None
                  else pd.Series(pd.NA, index=df.index, dtype="string"))
    return out


def _reconstruct(r: pd.DataFrame) -> pd.DataFrame:
    """Fill missing raw fields from exact identities (01_profile.md section 5). Row-wise only."""
    r = r.copy()
    fixed = r[["asst_intang_fixed", "asst_tang_fixed", "asst_fixed_fin"]].sum(axis=1, min_count=3)

    def fill(col, val):
        m = r[col].isna() & val.notna()
        r.loc[m, col] = val[m]

    fill("debt_st", r["asst_current"] - r["wc_net"])
    fill("asst_current", r["wc_net"] + r["debt_st"])
    fill("wc_net", r["asst_current"] - r["debt_st"])
    fill("eqty_tot", r["margin_fin"] + fixed)
    # total assets: components sum is a lower bound (omitted residual item, median 0.2%)
    comp = fixed + r["asst_current"]
    bad_ta = r["asst_tot"].isna() | (r["asst_tot"] <= 0)
    r.loc[bad_ta, "asst_tot"] = comp.where(comp > 0)[bad_ta]
    fill("prof_operations", r["rev_operating"] - r["COGS"])
    fill("prof_operations", r["roa"] * r["asst_tot"] / 100.0)
    fill("rev_operating", r["prof_operations"] + r["COGS"])
    fill("COGS", r["rev_operating"] - r["prof_operations"])
    fill("profit", r["cf_operations"] - r["ebitda"] + r["prof_operations"])
    fill("profit", r["roe"] * r["eqty_tot"] / 100.0)
    fill("ebitda", r["cf_operations"] - r["profit"] + r["prof_operations"])
    fill("cf_operations", r["profit"] + r["ebitda"] - r["prof_operations"])
    return r


# ---------------------------------------------------------------------------------------------
# Ratios
# ---------------------------------------------------------------------------------------------
def slog(v):
    """Signed log, sign(x) * ln(1 + |x|): monotone, so AUC and tree splits are unchanged; it
    stops a handful of firms with tiny denominators from dominating a linear model."""
    return np.sign(v) * np.log1p(np.abs(v))


def _div_ta(num: pd.Series, ta: pd.Series) -> pd.Series:
    return num / ta.where(ta > 0)


def _raw_features(r: pd.DataFrame) -> pd.DataFrame:
    """Unwinsorised features; may contain +/-inf by documented rules (clipped later)."""
    ta = r["asst_tot"]
    f = pd.DataFrame(index=r.index)
    f["log_assets"] = np.log(ta.where(ta > 0))
    f["equity_ratio"] = _div_ta(r["eqty_tot"], ta)
    f["st_debt_ratio"] = _div_ta(r["debt_st"], ta)
    f["bank_debt_ratio"] = _div_ta(r["debt_bank_st"] + r["debt_bank_lt"], ta)
    debt = r["debt_st"] + r["debt_lt"]
    f["st_debt_share"] = r["debt_st"] / debt.where(debt > 0)
    f["wc_ratio"] = _div_ta(r["wc_net"], ta)

    dst = r["debt_st"]
    cr = r["asst_current"] / dst.where(dst > 0)
    cr = cr.mask((dst == 0) & r["asst_current"].notna(), np.inf)
    f["current_ratio"] = cr
    f["cash_ratio"] = _div_ta(r["cash_and_equiv"], ta)

    f["roa_net"] = _div_ta(r["profit"], ta)
    f["roa_oper"] = _div_ta(r["prof_operations"], ta)
    rev = r["rev_operating"]
    f["ebitda_margin"] = r["ebitda"] / rev.where(rev > 0)

    ex, eb = r["exp_financing"], r["ebitda"]
    ic = eb / ex.where(ex > 0)
    noint = (ex <= 0) & eb.notna()
    ic = ic.mask(noint & (eb > 0), np.inf).mask(noint & (eb <= 0), 0.0)
    f["interest_coverage"] = ic

    cfo = r["cf_operations"]
    cd = cfo / debt.where(debt > 0)
    nodebt = (debt == 0) & cfo.notna()
    cd = cd.mask(nodebt & (cfo > 0), np.inf).mask(nodebt & (cfo <= 0), 0.0)
    f["cfo_to_debt"] = cd
    f["interest_to_assets"] = _div_ta(ex, ta)

    f["asset_turnover"] = _div_ta(rev, ta)
    ard = 365.0 * r["AR"] / rev.where(rev > 0)
    norev = (rev == 0) & r["AR"].notna()
    ard = ard.mask(norev & (r["AR"] > 0), np.inf).mask(norev & (r["AR"] <= 0), 0.0)
    f["ar_days"] = ard
    cogs = r["COGS"]
    f["ap_days"] = 365.0 * r["AP_st"] / cogs.where(cogs > 0)

    f["tangible_ratio"] = _div_ta(r["asst_tang_fixed"], ta)
    f["intangible_ratio"] = _div_ta(r["asst_intang_fixed"], ta)
    f["extraord_to_assets"] = _div_ta(r["inc_extraord"], ta)
    return f


def _flags(r: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=r.index)
    f["neg_equity"] = (r["eqty_tot"] < 0).astype("float64")
    f["debt_split_missing"] = r[["debt_bank_st", "debt_bank_lt", "AP_st"]].isna().any(axis=1).astype("float64")
    f["rev_nonpositive"] = (r["rev_operating"] <= 0).astype("float64")
    return f


def _map_unique(codes: pd.Series, fn) -> pd.Series:
    s = codes.astype(object).where(codes.notna(), None)
    lut = {u: fn(u) for u in pd.unique(s) if u is not None}
    return s.map(lambda c: lut.get(c) if c is not None else None)


def _section(codes: pd.Series) -> pd.Series:
    return _map_unique(codes, ateco_section)


def _macro(codes: pd.Series) -> pd.Series:
    def m(c):
        reg = province_region(c)
        return REGION_MACRO.get(reg) if reg else None
    return _map_unique(codes, m)


def _lookup(grp: pd.Series, tab: dict) -> pd.Series:
    """Per-row value from a {group: value} table with a '__global__' fallback."""
    return grp.map(tab).astype(float).fillna(tab["__global__"])


def _cats_raw(r: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "ateco_group": _section(r["ateco_sector"]),
        "legal_group": r["legal_struct"].astype(object).where(r["legal_struct"].notna(), None),
        "macro_area": _macro(r["HQ_city"]),
    }, index=r.index)


# ---------------------------------------------------------------------------------------------
# fit / transform
# ---------------------------------------------------------------------------------------------
def _finite_quantiles(s: pd.Series, qs) -> list[float]:
    v = s.replace([np.inf, -np.inf], np.nan).dropna()
    return [float(v.quantile(q)) for q in qs] if len(v) else [float("nan"), float("nan")]


def fit(train_df: pd.DataFrame, label_col: str = "y", feature_set: str = "full",
        id_col: str = "id") -> dict:
    """Learn all feature parameters from the training rows passed in (and nothing else).

    feature_set="full" (default) gives exactly the v1 params. feature_set="full2" additionally
    fits the round-2 parameters (params["v2"]: sector percentile grids, shrunk default-rate
    encodings incl. out-of-fold tables grouped by `id_col`, v2 winsor bounds and medians).
    """
    r = _reconstruct(coerce_input(train_df))
    raw = _raw_features(r)
    params: dict[str, Any] = {"version": VERSION, "n_train_rows": int(len(r)),
                              "winsor_q": list(WINSOR_Q), "winsor": {}, "categories": {},
                              "reference_level": {}, "impute_median": {}, "block_share": {}}
    if "fs_year" in train_df.columns:
        yrs = pd.to_numeric(train_df["fs_year"], errors="coerce").dropna().astype(int)
        params["train_years"] = sorted(int(x) for x in yrs.unique())
    for c in NUM_FEATURES:
        params["winsor"][c] = _finite_quantiles(raw[c], WINSOR_Q)

    # categoricals: pool levels with < MIN_DEFAULTS_PER_LEVEL train defaults
    y = pd.to_numeric(train_df[label_col], errors="coerce") if label_col in train_df.columns else None
    cats = _cats_raw(r)
    for c in CAT_FEATURES:
        s = cats[c].fillna(OTHER)
        if y is not None:
            d = y.groupby(s.values).sum()
            keep = sorted(k for k, v in d.items() if v >= MIN_DEFAULTS_PER_LEVEL and k != OTHER)
            params["min_count_rule"] = f">= {MIN_DEFAULTS_PER_LEVEL} training defaults"
        else:  # no label: fall back to a row-count rule (>= 1% of rows)
            n = s.value_counts()
            keep = sorted(k for k, v in n.items() if v >= 0.01 * len(s) and k != OTHER)
            params["min_count_rule"] = ">= 1% of training rows (no label supplied)"
        params["categories"][c] = keep + [OTHER]
        s2 = s.where(s.isin(keep), OTHER)
        params["reference_level"][c] = str(s2.value_counts().idxmax())
    grp = cats["ateco_group"].fillna(OTHER).where(cats["ateco_group"].isin(params["categories"]["ateco_group"]), OTHER)

    # debt-split block: sector median share of each component in its total
    for comps, tot in ((ST_BLOCK, "debt_st"), (LT_BLOCK, "debt_lt")):
        present = r[comps].notna().all(axis=1) & (r[tot] > 0)
        for c in comps:
            share = (r[c] / r[tot]).clip(0, 1)[present]
            g = share.groupby(grp[present]).median()
            params["block_share"][c] = {"__global__": float(share.median()),
                                        **{str(k): float(v) for k, v in g.items()}}

    # imputation medians: on the (block-imputed, winsorised) train features, by sector group
    X = transform(train_df, params, impute=False, _clean=r)
    Xb = transform(train_df, params, impute=False, _clean=_impute_block(r, grp, params))
    for c in NUM_FEATURES:
        src = Xb[c] if c in BLOCK_FEATURES else X[c]
        g = src.groupby(grp).median()
        params["impute_median"][c] = {"__global__": float(src.median()),
                                      **{str(k): float(v) for k, v in g.items() if pd.notna(v)}}
    if feature_set == "full2":
        params["v2"] = _fit_v2(train_df, r, grp, params, label_col, id_col)
    elif feature_set not in ("full", "core"):
        raise ValueError(f"unknown feature_set {feature_set!r}")
    return params


def _impute_block(r: pd.DataFrame, grp: pd.Series, params: dict) -> pd.DataFrame:
    r = r.copy()
    for comps, tot in ((ST_BLOCK, "debt_st"), (LT_BLOCK, "debt_lt")):
        for c in comps:
            tab = params["block_share"][c]
            sh = _lookup(grp, tab)
            m = r[c].isna() & r[tot].notna()
            r.loc[m, c] = (sh * r[tot].clip(lower=0))[m]
    return r


def transform(df: pd.DataFrame, params: dict, impute: bool = False, _clean=None,
              feature_set: str = "full", oof: bool = False, id_col: str = "id") -> pd.DataFrame:
    """Features for every input row, same order and index. Never uses statistics of `df`.

    feature_set: "full" (default, v1 columns ALL_FEATURES, unchanged) or "full2" (FULL2_FEATURES;
    needs params fitted with fit(..., feature_set="full2")). "core" returns the full v1 frame too
    (select CORE_FEATURES yourself, as before).
    oof=True is for TRAINING rows only: the shrunk default-rate encodings are then taken from the
    out-of-fold table of the row's firm (fold = crc32(firm id) mod K, stored in params), so a
    training row never sees its own label. The harness always uses oof=False.
    """
    c0 = coerce_input(df) if _clean is None else None
    r = _reconstruct(c0) if _clean is None else _clean
    r_pre = r
    cats = _cats_raw(r)
    out = pd.DataFrame(index=r.index)
    for c in CAT_FEATURES:
        lv = params["categories"][c]
        s = cats[c].where(cats[c].isin(lv), OTHER).fillna(OTHER)
        out[c] = pd.Categorical(s, categories=lv)
    grp = out["ateco_group"].astype(str)

    flags = _flags(r)
    if impute:
        r = _impute_block(r, grp, params)
    raw = _raw_features(r)
    for c in NUM_FEATURES:
        lo, hi = params["winsor"][c]
        v = raw[c].clip(lower=lo, upper=hi)          # +/-inf land on the bounds, NaN stays NaN
        if META[c].get("slog"):
            v = slog(v)                              # monotone: same ranks, tamer tails for GLMs
        if impute:
            tab = params["impute_median"][c]
            fillv = _lookup(grp, tab)
            v = v.fillna(fillv)
        out[c] = v.astype("float64")
    for c in FLAG_FEATURES:
        out[c] = flags[c]
    if feature_set in ("full2", "full2_candidates"):
        if "v2" not in params:
            raise ValueError('params lack "v2": fit with feature_set="full2"')
        v2 = _transform_v2(df, c0 if c0 is not None else r_pre, r_pre, r, out, params, impute,
                           oof, id_col)
        cols = FULL2_FEATURES if feature_set == "full2" else ALL_FEATURES + V2_CANDIDATES
        return pd.concat([out[ALL_FEATURES], v2], axis=1)[cols]   # candidates: evidence scripts only
    return out[ALL_FEATURES]


# ---------------------------------------------------------------------------------------------
# helpers for challengers
# ---------------------------------------------------------------------------------------------
def design_matrix(X: pd.DataFrame, params: dict, features: list[str] | None = None) -> pd.DataFrame:
    """Float matrix for GLMs: numeric + flags as is, categoricals one-hot with the training
    reference level (largest level) dropped. Column set is fixed by params, not by the data."""
    features = features or ALL_FEATURES
    parts = [X[[c for c in features if c not in CAT_FEATURES]].astype("float64")]
    for c in [c for c in features if c in CAT_FEATURES]:
        ref = params["reference_level"][c]
        for lv in params["categories"][c]:
            if lv == ref:
                continue
            parts.append(pd.Series((X[c].astype(str) == lv).astype("float64"),
                                   index=X.index, name=f"{c}={lv}"))
    return pd.concat(parts, axis=1)


def expected_signs(features: list[str] | None = None) -> dict[str, int]:
    """Expected direction vs PD (+1, -1, 0). Use for monotone constraints / face validity."""
    features = features or (NUM_FEATURES + FLAG_FEATURES)
    return {c: int(META[c]["sign"]) for c in features if c in META}


def save_params(params: dict, path: str) -> None:
    with open(path, "w") as fh:
        json.dump(params, fh, indent=1, sort_keys=True)


def load_params(path: str) -> dict:
    with open(path) as fh:
        return json.load(fh)


# =============================================================================================
# ROUND 2 (features-v2, feature-architect 2026-11-02). Opt-in only: fit(..., feature_set="full2")
# and transform(..., feature_set="full2"). Nothing above this line changes its output.
# Decisions: reports/features/round2_decisions.md (D-090..D-099);
# evidence: reports/features/round2_univariate.md (training rows fs_year 2007-2009 only).
# =============================================================================================
import zlib  # noqa: E402

VERSION_V2 = "features-v2 (2026-11-02)"
TE_FOLDS = 5                    # out-of-fold folds for the default-rate encodings (grouped by firm)
TE_M_BOUNDS = (50.0, 50000.0)   # guard rails on the empirical-Bayes prior strength m (rows)
PCT_MIN_ROWS = 2000             # an ATECO section needs this many training rows for its own grid
PCT_GRID = [i / 100 for i in range(101)]
PCT_BASE = ["equity_ratio", "roa_net", "interest_coverage", "cash_ratio", "st_debt_ratio",
            "tangible_ratio"]
MISS_COUNT_FIELDS = [c for c in RAW_NUM if c not in ("roe", "margin_fin")]
CORE_NUM = [c for c in CORE_FEATURES if c in NUM_FEATURES]

_R33 = "R-33 Table 3 p. 11 (input), p. 12 (direction of the Fig. 5 transform)"
V2_FEATURES: list[dict[str, Any]] = [
    # ---- F1: RiskCalc 4.0 France ratios -----------------------------------------------------
    dict(name="net_debt_coverage", family="coverage", sign=-1, slog=True, src=_R33, rejected=True,
         formula="slog(cf_operations / ((asst_tot - eqty_tot) - cash_and_equiv))", units="1/yr",
         reason="Cash earnings (net profit + D&A) per euro of liabilities not already covered by "
                "cash: the inverse of the years needed to pay off net liabilities. RiskCalc France's "
                "heaviest-weighted input (19.2%, R-33 Table 4 p. 14).",
         zero_rule="net liabilities <= 0 (cash covers every liability: the safest case) -> +inf if "
                   "cf_operations > 0, else 0 (no cash generation); +inf clipped to the train p99",
         missing_rule="NaN if any input missing after identity reconstruction; impute: sector median"),
    dict(name="operating_margin", family="profitability", sign=-1, slog=True, src=_R33,
         formula="slog(prof_operations / rev_operating)", units="share of sales",
         reason="Operating profit per euro of sales: the buffer before costs exceed revenue. "
                "RiskCalc France's profitability input (replaces EBITDA/sales).",
         zero_rule="rev_operating <= 0 -> NaN (rev_nonpositive flag marks it)",
         missing_rule="NaN; impute: sector median"),
    dict(name="interest_to_sales", family="coverage/activity", sign=0, slog=True, src=_R33 +
         "; R-33 shows an upward slope, our 2007-09 deciles are hump-shaped (D-092), so sign none",
         formula="slog(exp_financing / rev_operating)", units="share of sales",
         reason="Share of every sales euro already spent on interest: high values leave little "
                "room before a payment is missed.",
         zero_rule="rev_operating == 0: +inf if exp_financing > 0 else 0; rev_operating < 0 -> NaN",
         missing_rule="NaN; impute: sector median"),
    dict(name="payables_to_sales", family="activity", sign=+1, slog=True, src=_R33, block=True,
         formula="slog(AP_st / rev_operating)", units="share of sales",
         reason="Large trade payables relative to sales = suppliers are being stretched, a classic "
                "sign of cash stress (R-33 p. 10, Table 2 gives exactly this example).",
         zero_rule="rev_operating == 0: +inf if AP_st > 0 else 0; rev_operating < 0 -> NaN",
         missing_rule="NaN if the debt-split block is missing; impute: sector share x debt_st, "
                      "then sector median"),
    dict(name="personnel_to_sales", family="activity", sign=+1, slog=True, src=_R33 +
         "; field meaning = D-010 hypothesis (goodwill = value added), tested in D-091",
         formula="slog((goodwill - ebitda) / rev_operating)   [HYPOTHESIS-DEPENDENT]",
         units="share of sales",
         reason="Staff costs are hard to cut quickly; a high wage bill per euro of sales means a "
                "drop in sales turns straight into losses (operating leverage).",
         zero_rule="rev_operating == 0: +inf if cost > 0 else 0; rev_operating < 0 -> NaN; "
                   "goodwill < ebitda (negative personnel cost, 0.01% of rows) -> NaN",
         missing_rule="NaN; impute: sector median"),
    # ---- F2: sector-relative percentiles ---------------------------------------------------
    *[dict(name=f"{b}_sect_pct", family="sector-relative", sign=META[b]["sign"], kind="pct",
           src="R-01 Sec. 3.3 pp. 17-18 (industry corrects ratios whose meaning differs by sector)",
           formula=f"percentile of {b} within the firm's ATECO section (train grid)",
           units="0-1",
           reason=f"Where the firm stands among peers of its own sector on {b}: the same ratio "
                  "means different things in, e.g., real estate and retail.",
           zero_rule="inherits the rule of the base feature; values outside the grid -> 0 or 1",
           missing_rule="NaN if the base is NaN; impute: computed from the imputed base "
                        f"(sector median -> about 0.5); sections with < {PCT_MIN_ROWS} train rows "
                        "or unseen -> pooled grid")
      for b in PCT_BASE],
    # ---- F3: shrunk default-rate encodings -------------------------------------------------
    dict(name="ateco_div_te", family="industry", sign=+1, kind="te",
         src="assumption (empirical-Bayes / m-estimate encoding; no VERIFIED source in REFERENCES)",
         formula="logit((d_l + m p0)/(n_l + m)) - logit(p0), l = ATECO division (2-digit)",
         units="log-odds vs base rate",
         reason="The division's own historical default rate, pulled toward the average when the "
                "division is small: finer than the 21 sections, without 83 dummies.",
         zero_rule="n/a", missing_rule="missing / unseen division -> 0 (= the base rate)"),
    dict(name="province_te", family="geography", sign=+1, kind="te",
         src="assumption (as ateco_div_te)",
         formula="same as ateco_div_te, l = HQ_city (province)", units="log-odds vs base rate",
         reason="Local economic conditions (e.g. local banks, courts, demand) differ by province.",
         zero_rule="n/a", missing_rule="missing / unseen province -> 0"),
    dict(name="region_te", family="geography", sign=+1, kind="te",
         src="assumption (as ateco_div_te)",
         formula="same as ateco_div_te, l = region of HQ_city", units="log-odds vs base rate",
         reason="Regional conditions; 20 levels, so less noisy than province.",
         zero_rule="n/a", missing_rule="missing / unseen region -> 0"),
    # ---- F4: data quality / missingness ----------------------------------------------------
    dict(name="n_missing_raw", family="data quality", sign=+1, kind="count", rejected=True,
         src="R-34 / R-05 pp. 60-61 (less real data, worse/biased estimate; L1)",
         formula=f"count of missing raw amounts among {len(MISS_COUNT_FIELDS)} fields "
                 "(roe, margin_fin excluded: blank by construction when equity <= 0)",
         units="count",
         reason="A thinner statement (abbreviated filing, incomplete data) tells the bank less "
                "and in our data comes with slightly higher default rates.",
         zero_rule="n/a", missing_rule="never missing"),
    dict(name="n_core_imputed", family="data quality", sign=+1, kind="count",
         src="R-34 (L1: median fill understates PD)",
         formula="count of core numeric features still NaN after identity reconstruction "
                 "(= the number the GLM view fills with a sector median)", units="count",
         reason="Flags rows whose PD rests partly on sector medians rather than the firm's own "
                "numbers, so the model can add back the risk a median hides.",
         zero_rule="n/a", missing_rule="never missing"),
    dict(name="bs_gap", family="data quality", sign=0,
         src="assumption; data-auditor 01_profile.md section 5 (omitted liability item)",
         formula="(asst_tot - eqty_tot - debt_st - debt_lt - liab_lt - liab_lt_emp) / asst_tot",
         units="share of assets",
         reason="Liabilities the file does not classify (most likely accruals / deferred income "
                "[hypothesis]). Large gaps can mean unusual balance sheets or poor data; no "
                "monotone expectation.",
         zero_rule="denominator is total assets (> 0)", missing_rule="NaN; impute: sector median"),
    # ---- F5: unused fields -----------------------------------------------------------------
    dict(name="nonbank_fin_debt", family="leverage", sign=+1, block=True, src="assumption",
         formula="(debt_fin_st + debt_fin_lt) / asst_tot", units="share of assets",
         reason="Financial debt owed to non-banks (leasing, factoring, other lenders, shareholder "
                "loans): more fixed claims on the same assets.",
         zero_rule="denominator is total assets (> 0)",
         missing_rule="NaN if the block is missing; impute: sector share x debt totals, then median"),
    dict(name="ap_lt_ratio", family="leverage", sign=+1, block=True, src="assumption",
         formula="AP_lt / asst_tot", units="share of assets",
         reason="Trade payables due after more than a year are unusual: suppliers converted into "
                "long-term creditors, often after a renegotiation.",
         zero_rule="denominator is total assets (> 0)",
         missing_rule="NaN if the block is missing; impute: sector share x debt_lt, then median"),
    dict(name="fin_fixed_ratio", family="asset structure", sign=0, src="assumption",
         formula="asst_fixed_fin / asst_tot", units="share of assets",
         reason="Equity stakes and loans to other firms: holding / group structure. Group support "
                "can help or the stakes can be illiquid and tied to group trouble: no monotone sign.",
         zero_rule="denominator is total assets (> 0)", missing_rule="NaN; impute: sector median"),
    dict(name="tfr_ratio", family="asset structure", sign=0, src="assumption",
         formula="liab_lt_emp / asst_tot  (TFR staff-severance fund)", units="share of assets",
         reason="Proxy for labour intensity (and an internal funding source owed to staff). Two "
                "opposite stories, so no monotone sign.",
         zero_rule="denominator is total assets (> 0)", missing_rule="NaN; impute: sector median"),
]
V2_NUM = [f["name"] for f in V2_FEATURES if f.get("kind", "num") == "num"]
V2_PCT = [f["name"] for f in V2_FEATURES if f.get("kind") == "pct"]
V2_TE = [f["name"] for f in V2_FEATURES if f.get("kind") == "te"]
V2_COUNT = [f["name"] for f in V2_FEATURES if f.get("kind") == "count"]
V2_BLOCK = [f["name"] for f in V2_FEATURES if f.get("block")]
V2_CANDIDATES = [f["name"] for f in V2_FEATURES]            # everything evaluated in round 2
V2_REJECTED = [f["name"] for f in V2_FEATURES if f.get("rejected")]   # computed, not in the set
NEW_FEATURES_V2 = [c for c in V2_CANDIDATES if c not in V2_REJECTED]
FULL2_FEATURES = ALL_FEATURES + NEW_FEATURES_V2
FEATURE_SETS = {"core": CORE_FEATURES, "full": ALL_FEATURES, "full2": FULL2_FEATURES}
for _f in V2_FEATURES:                       # add (never overwrite) metadata for expected_signs()
    assert _f["name"] not in META
    META[_f["name"]] = {"kind": "num", "core": False, **_f}


def _rev_ratio(num: pd.Series, rev: pd.Series) -> pd.Series:
    """num / revenue with the ar_days rule: rev == 0 -> +inf if num > 0 else 0; rev < 0 -> NaN."""
    v = num / rev.where(rev > 0)
    norev = (rev == 0) & num.notna()
    return v.mask(norev & (num > 0), np.inf).mask(norev & (num <= 0), 0.0)


def _raw_v2(r: pd.DataFrame) -> pd.DataFrame:
    ta, rev = r["asst_tot"], r["rev_operating"]
    f = pd.DataFrame(index=r.index)
    cfo = r["cf_operations"]
    netl = (ta - r["eqty_tot"]) - r["cash_and_equiv"]
    v = cfo / netl.where(netl > 0)
    netcash = (netl <= 0) & cfo.notna()
    f["net_debt_coverage"] = v.mask(netcash & (cfo > 0), np.inf).mask(netcash & (cfo <= 0), 0.0)
    f["operating_margin"] = r["prof_operations"] / rev.where(rev > 0)
    f["interest_to_sales"] = _rev_ratio(r["exp_financing"], rev)
    f["payables_to_sales"] = _rev_ratio(r["AP_st"], rev)
    pc = r["goodwill"] - r["ebitda"]
    f["personnel_to_sales"] = _rev_ratio(pc.where(pc >= 0), rev)
    f["bs_gap"] = _div_ta(ta - r["eqty_tot"] - r["debt_st"] - r["debt_lt"] - r["liab_lt"]
                          - r["liab_lt_emp"], ta)
    f["nonbank_fin_debt"] = _div_ta(r["debt_fin_st"] + r["debt_fin_lt"], ta)
    f["ap_lt_ratio"] = _div_ta(r["AP_lt"], ta)
    f["fin_fixed_ratio"] = _div_ta(r["asst_fixed_fin"], ta)
    f["tfr_ratio"] = _div_ta(r["liab_lt_emp"], ta)
    return f[V2_NUM]


def _firm_fold(ids: pd.Series, k: int) -> np.ndarray:
    """Deterministic firm -> fold map (crc32 of the normalised id string); no randomness to store."""
    codes = _to_code(ids).fillna("")
    lut = {u: zlib.crc32(str(u).encode()) % k for u in pd.unique(codes)}
    return codes.map(lut).to_numpy(dtype=int)


def _te_levels(r: pd.DataFrame) -> dict[str, pd.Series]:
    def reg(c):
        return province_region(c)
    prov = r["HQ_city"].astype(object).where(r["HQ_city"].notna(), None)
    div = r["ateco_sector"].astype(object).where(r["ateco_sector"].notna(), None)
    return {"ateco_div_te": div, "province_te": prov, "region_te": _map_unique(r["HQ_city"], reg)}


def _logit(p):
    return float(np.log(p / (1 - p)))


def _te_table(lv: pd.Series, y: pd.Series) -> dict:
    """Shrunk log-odds encoding. Prior strength m by empirical Bayes (method of moments on the
    between-level variance of default rates), clipped to TE_M_BOUNDS."""
    ok = lv.notna() & y.notna()
    g = y[ok].groupby(lv[ok].values).agg(["sum", "count"])
    n_tot, d_tot = float(g["count"].sum()), float(g["sum"].sum())
    p0 = min(max(d_tot / n_tot, 1e-6), 1 - 1e-6)
    n, rate = g["count"].astype(float), g["sum"] / g["count"]
    L = len(g)
    s_between = float((n * (rate - p0) ** 2).sum())
    denom = n_tot - float((n ** 2).sum()) / n_tot
    tau2 = (s_between - (L - 1) * p0 * (1 - p0)) / denom if denom > 0 else 0.0
    m = p0 * (1 - p0) / tau2 if tau2 > 0 else TE_M_BOUNDS[1]
    m = float(np.clip(m, *TE_M_BOUNDS))
    ps = (g["sum"] + m * p0) / (g["count"] + m)
    enc = {str(k): float(np.log(v / (1 - v)) - _logit(p0)) for k, v in ps.items()}
    return {"m": m, "p0": p0, "tau2": float(tau2), "n_levels": L, "enc": enc}


def _pct_grid(v: pd.Series) -> list[float]:
    v = v.replace([np.inf, -np.inf], np.nan).dropna()
    return [float(x) for x in np.quantile(v.to_numpy(), PCT_GRID)] if len(v) else []


def _pct_lookup(x: pd.Series, grid: list[float]) -> pd.Series:
    """Percentile of x on a stored quantile grid; ties in the grid get their mid-percentile."""
    if not grid:
        return pd.Series(np.nan, index=x.index)
    g = pd.Series(PCT_GRID, index=np.asarray(grid)).groupby(level=0).mean()
    xv = x.to_numpy(dtype=float)
    out = np.interp(xv, g.index.to_numpy(), g.to_numpy(), left=0.0, right=1.0)
    return pd.Series(np.where(np.isnan(xv), np.nan, out), index=x.index)


def _fit_v2(train_df, r, grp, params, label_col, id_col) -> dict:
    v2: dict[str, Any] = {"version": VERSION_V2, "winsor": {}, "impute_median": {}, "pct": {},
                          "te": {}, "te_folds": TE_FOLDS, "pct_min_rows": PCT_MIN_ROWS}
    raw = _raw_v2(r)
    for c in V2_NUM:
        v2["winsor"][c] = _finite_quantiles(raw[c], WINSOR_Q)
    # sector percentile grids on the v1 (winsorised, non-imputed) training features
    X = transform(train_df, params, impute=False, _clean=r)
    sect = _section(r["ateco_sector"])
    for b in PCT_BASE:
        tab = {"__global__": _pct_grid(X[b])}
        for s_, idx in X[b].groupby(sect.fillna(OTHER).values).groups.items():
            if s_ != OTHER and len(idx) >= PCT_MIN_ROWS:
                tab[str(s_)] = _pct_grid(X[b].loc[idx])
        v2["pct"][b] = tab
    # shrunk default-rate encodings: full window (for scoring) + out-of-fold tables (training)
    if label_col not in train_df.columns:
        raise ValueError("feature_set='full2' needs the label column to fit the encodings")
    y = pd.to_numeric(train_df[label_col], errors="coerce").reset_index(drop=True)
    levels = {k: v.reset_index(drop=True) for k, v in _te_levels(r).items()}
    fold = _firm_fold(train_df[id_col].reset_index(drop=True), TE_FOLDS) if id_col in train_df.columns else None
    for name, lv in levels.items():
        ent = {"full": _te_table(lv, y)}
        if fold is not None:
            ent["oof"] = [_te_table(lv[fold != k], y[fold != k]) for k in range(TE_FOLDS)]
        v2["te"][name] = ent
    # medians for the GLM view (block features: from the block-imputed rows)
    rb = _impute_block(r, grp, params)
    raw_b = _raw_v2(rb)
    for c in V2_NUM:
        src = raw_b[c] if c in V2_BLOCK else raw[c]
        lo, hi = v2["winsor"][c]
        src = src.clip(lower=lo, upper=hi)
        if META[c].get("slog"):
            src = slog(src)
        g = src.groupby(grp).median()
        v2["impute_median"][c] = {"__global__": float(src.median()),
                                  **{str(k): float(v) for k, v in g.items() if pd.notna(v)}}
    return v2


def _transform_v2(df, c0, r_pre, r, out, params, impute, oof, id_col) -> pd.DataFrame:
    v2 = params["v2"]
    grp = out["ateco_group"].astype(str)
    res = pd.DataFrame(index=r.index)
    raw = _raw_v2(r)                      # r is block-imputed when impute=True
    for c in V2_NUM:
        lo, hi = v2["winsor"][c]
        v = raw[c].clip(lower=lo, upper=hi)
        if META[c].get("slog"):
            v = slog(v)
        if impute:
            v = v.fillna(_lookup(grp, v2["impute_median"][c]))
        res[c] = v.astype("float64")
    sect = _section(r_pre["ateco_sector"])
    for b in PCT_BASE:
        tab = v2["pct"][b]
        val = pd.Series(np.nan, index=r.index)
        key = sect.where(sect.isin([k for k in tab if k != "__global__"]), "__global__")
        for k in pd.unique(key):
            m = (key == k).to_numpy()
            val[m] = _pct_lookup(out[b][m], tab[k]).to_numpy()
        res[f"{b}_sect_pct"] = val
    levels = _te_levels(r_pre)
    if oof:
        if id_col not in df.columns:
            raise ValueError("oof=True needs the firm id column (training rows only)")
        fold = _firm_fold(pd.Series(np.asarray(df[id_col]), index=r.index), v2["te_folds"])
    for name in V2_TE:
        lv = levels[name]
        if oof:
            val = pd.Series(0.0, index=r.index)
            for k in range(v2["te_folds"]):
                m = fold == k
                enc = v2["te"][name]["oof"][k]["enc"]
                val[m] = lv[m].map(enc).astype(float).fillna(0.0).to_numpy()
        else:
            val = lv.map(v2["te"][name]["full"]["enc"]).astype(float).fillna(0.0)
        res[name] = val.astype("float64")
    res["n_missing_raw"] = c0[MISS_COUNT_FIELDS].isna().sum(axis=1).astype("float64")
    res["n_core_imputed"] = _raw_features(r_pre)[CORE_NUM].replace(
        [np.inf, -np.inf], 0.0).isna().sum(axis=1).astype("float64")
    return res[V2_CANDIDATES]
