"""Unit tests for src/features/build.py (feature-architect, 2026-10-30).

Run: python -m pytest src/features/test_build.py -q
Synthetic data only (fast, no panel needed), plus one optional check against the panel.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pytest

from src.features import build as B

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _synthetic(n=3000, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ta = np.exp(rng.normal(15.5, 1.0, n)) + 1.5e6
    ca = ta * rng.uniform(0.2, 0.8, n)
    tang = ta * rng.uniform(0.0, 0.5, n)
    intang = ta * rng.uniform(0.0, 0.05, n)
    fin = ta - ca - tang - intang
    eq = ta * rng.normal(0.25, 0.2, n)
    dst = ta * rng.uniform(0.2, 0.7, n)
    dlt = np.clip(ta - eq - dst, 0, None)
    rev = ta * rng.uniform(0.0, 2.5, n)
    cogs = rev * rng.uniform(0.8, 1.05, n)
    po = rev - cogs
    ebitda = po + ta * 0.02
    exp_fin = np.abs(dst + dlt) * 0.04
    profit = po - exp_fin
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(1 + 4 * eq / ta))).astype(float)  # ~12% so levels keep >= 30 defaults
    df = pd.DataFrame({
        "Unnamed: 0": np.arange(n), "id": np.arange(n), "fs_year": 2008,
        "HQ_city": rng.choice(["15", "58", "63", "90", "1"], n),
        "legal_struct": rng.choice(["SRL", "SPA", "SRU", "SAU"], n),
        "ateco_sector": rng.choice(["28", "46", "41", "68", "10"], n),
        "asst_intang_fixed": intang, "asst_tang_fixed": tang, "asst_fixed_fin": fin,
        "asst_current": ca, "AR": ca * 0.5, "cash_and_equiv": ca * 0.1, "asst_tot": ta,
        "eqty_tot": eq, "eqty_corp_family_tot": np.nan, "liab_lt": 0.0, "liab_lt_emp": ta * 0.01,
        "debt_bank_st": dst * 0.3, "debt_bank_lt": dlt * 0.6, "debt_fin_st": 0.0, "debt_fin_lt": 0.0,
        "AP_st": dst * 0.5, "AP_lt": 0.0, "debt_st": dst, "debt_lt": dlt,
        "rev_operating": rev, "COGS": cogs, "prof_operations": po, "goodwill": ebitda * 2,
        "inc_financing": -exp_fin, "exp_financing": exp_fin, "prof_financing": 0.0,
        "inc_extraord": 0.0, "taxes": profit, "profit": profit, "days_rec": np.nan,
        "ebitda": ebitda, "roa": 100 * po / ta, "roe": 100 * profit / eq, "wc_net": ca - dst,
        "margin_fin": eq - tang - intang - fin, "cf_operations": profit + (ebitda - po), "y": y,
    })
    # a block-missing chunk, like abbreviated statements
    blk = rng.uniform(size=n) < 0.12
    df.loc[blk, ["debt_bank_st", "debt_fin_st", "AP_st", "debt_bank_lt", "debt_fin_lt", "AP_lt"]] = np.nan
    return df


@pytest.fixture(scope="module")
def train():
    return _synthetic()


@pytest.fixture(scope="module")
def params(train):
    return B.fit(train)


def _check_output(X, n, params):
    assert len(X) == n
    assert list(X.columns) == B.ALL_FEATURES
    num = X[B.NUM_FEATURES + B.FLAG_FEATURES].to_numpy(float)
    assert not np.isinf(num).any(), "outputs must be finite or NaN"
    for c in B.CAT_FEATURES:
        assert X[c].notna().all()
        assert set(X[c].astype(str)) <= set(params["categories"][c])


def test_params_json_roundtrip(params, tmp_path):
    s = json.dumps(params)                      # must not raise
    p2 = json.loads(s)
    path = tmp_path / "p.json"
    B.save_params(params, str(path))
    assert B.load_params(str(path)) == p2


def test_transform_basic_and_order(train, params):
    X = B.transform(train, params)
    _check_output(X, len(train), params)
    assert X.index.equals(train.index)
    shuffled = train.sample(frac=1.0, random_state=1)
    Xs = B.transform(shuffled, params)
    assert Xs.index.equals(shuffled.index)
    pd.testing.assert_frame_equal(Xs.loc[train.index], X)


def test_single_row_equals_batch_row(train, params):
    """No batch statistics: a row's features do not depend on the other rows."""
    X = B.transform(train, params, impute=True)
    for i in [0, 7, 123]:
        xi = B.transform(train.iloc[[i]], params, impute=True)
        pd.testing.assert_frame_equal(xi, X.iloc[[i]])


def test_impute_fills_everything(train, params):
    Xi = B.transform(train, params, impute=True)
    assert Xi[B.NUM_FEATURES + B.FLAG_FEATURES].notna().all().all()
    Z = B.design_matrix(Xi, params)
    assert np.isfinite(Z.to_numpy(float)).all()


def test_messy_row(params):
    messy = pd.DataFrame([{
        "asst_tot": " 2,500,000 ", "eqty_tot": "NA", "margin_fin": "100000",
        "asst_intang_fixed": "0", "asst_tang_fixed": "n.a.", "asst_fixed_fin": "", "asst_current": "#DIV/0!",
        "debt_st": "0", "debt_lt": "-", "rev_operating": "0", "AR": "5000", "COGS": "abc",
        "exp_financing": "0", "ebitda": "-10", "profit": "inf", "cf_operations": None,
        "HQ_city": " 999 ", "legal_struct": "XYZ", "ateco_sector": "28.0",
    }])
    X = B.transform(messy, params)
    _check_output(X, 1, params)
    assert X.loc[0, "ateco_group"] == "C"
    assert X.loc[0, "legal_group"] == B.OTHER and X.loc[0, "macro_area"] == B.OTHER
    # zero-interest + negative EBITDA -> 0 coverage (documented rule), not inf / NaN
    assert X.loc[0, "interest_coverage"] == 0.0
    # zero revenue with AR > 0 -> upper bound of ar_days (on slog scale)
    assert X.loc[0, "ar_days"] == pytest.approx(B.slog(params["winsor"]["ar_days"][1]))
    assert X.loc[0, "rev_nonpositive"] == 1.0
    Xi = B.transform(messy, params, impute=True)
    assert Xi[B.NUM_FEATURES].notna().all().all()


def test_all_nan_row_and_empty_frame(params):
    row = pd.DataFrame([{c: np.nan for c in B.RAW_NUM + B.RAW_CAT}])
    X = B.transform(row, params)
    _check_output(X, 1, params)
    assert X[B.NUM_FEATURES].isna().all().all()
    Xi = B.transform(row, params, impute=True)
    assert Xi[B.NUM_FEATURES].notna().all().all()
    # a frame with no recognised columns at all
    X2 = B.transform(pd.DataFrame({"foo": [1, 2]}), params, impute=True)
    _check_output(X2, 2, params)
    X0 = B.transform(pd.DataFrame(columns=B.RAW_NUM + B.RAW_CAT), params)
    assert len(X0) == 0 and list(X0.columns) == B.ALL_FEATURES


def test_alias_columns_and_extra_index(train, params):
    df = train.drop(columns=["Unnamed: 0"]).rename(columns={
        "eqty_tot": "eqyt_tot", "rev_operating": "rev_operations", "prof_operations": "prof_operating"})
    df["extra_col"] = "zzz"
    X1 = B.transform(train, params)
    X2 = B.transform(df, params)
    pd.testing.assert_frame_equal(X1, X2)


def test_numeric_category_codes_and_strings_equal(train, params):
    df = train.copy()
    df["ateco_sector"] = df["ateco_sector"].astype(float)
    df["HQ_city"] = df["HQ_city"].astype(int)
    pd.testing.assert_frame_equal(B.transform(df, params), B.transform(train, params))


def test_infinite_and_zero_denominators(params, train):
    r = train.iloc[[0]].copy()
    r["debt_st"] = 0.0
    r["wc_net"] = r["asst_current"]
    r["exp_financing"] = 0.0
    r["ebitda"] = 1e6
    r["debt_lt"] = 0.0
    r["cf_operations"] = 1e5
    X = B.transform(r, params)
    hi = params["winsor"]
    assert X["current_ratio"].iloc[0] == pytest.approx(B.slog(hi["current_ratio"][1]))
    assert X["interest_coverage"].iloc[0] == pytest.approx(B.slog(hi["interest_coverage"][1]))
    assert X["cfo_to_debt"].iloc[0] == pytest.approx(hi["cfo_to_debt"][1])
    r2 = train.iloc[[0]].copy()
    r2["asst_current"] = np.inf
    r2["asst_tot"] = -np.inf          # -> rebuilt from components (which include NaN current) -> NaN
    X2 = B.transform(r2, params)
    assert not np.isinf(X2[B.NUM_FEATURES].to_numpy(float)).any()


def test_identity_reconstruction(params, train):
    r = train.iloc[[3]].copy()
    full = B.transform(r, params)
    r["eqty_tot"] = np.nan             # rebuilt from margin_fin + fixed assets
    r["debt_st"] = np.nan              # rebuilt from asst_current - wc_net
    X = B.transform(r, params)
    for c in ["equity_ratio", "st_debt_ratio", "current_ratio"]:
        assert X[c].iloc[0] == pytest.approx(full[c].iloc[0], rel=1e-6)


def test_fit_uses_only_given_rows(train):
    """Changing rows that are NOT passed to fit cannot change params (trivially true, but guards
    against accidental globals); and params differ when the training rows differ."""
    p1 = B.fit(train.iloc[:1500])
    p2 = B.fit(train.iloc[:1500].copy())
    assert json.dumps(p1, sort_keys=True) == json.dumps(p2, sort_keys=True)
    p3 = B.fit(train.iloc[1500:])
    assert p1["winsor"] != p3["winsor"]


def test_unseen_and_pooled_categories(params):
    assert params["categories"]["ateco_group"][-1] == B.OTHER
    df = pd.DataFrame({"ateco_sector": ["99", "85", None, "nan", "C"], "asst_tot": 2e6,
                       "HQ_city": ["1", "105", "0", "", "abc"], "legal_struct": ["SRS", "SAA", "", "srl", None]})
    X = B.transform(df, params)
    assert X["legal_group"].astype(str).tolist()[3] == "SRL"
    assert X["macro_area"].astype(str).tolist()[0] == "North-West"
    assert (X["macro_area"].astype(str).iloc[2:] == B.OTHER).all()


def test_expected_signs_cover_all_numeric():
    s = B.expected_signs()
    assert set(s) == set(B.NUM_FEATURES + B.FLAG_FEATURES)
    assert set(s.values()) <= {-1, 0, 1}
    assert 8 <= len([c for c in B.CORE_FEATURES if c not in B.CAT_FEATURES]) <= 12


@pytest.mark.skipif(not os.path.exists(os.path.join(ROOT, "data", "interim", "panel.parquet")),
                    reason="panel not available")
def test_panel_sample_runs():
    p = pd.read_parquet(os.path.join(ROOT, "data", "interim", "panel.parquet"))
    tr = p[(p["label_status"] == "ok") & (p["fs_year"] == 2007)].sample(20000, random_state=0)
    te = p[p["fs_year"] == 2012].sample(5000, random_state=0)   # unlabeled future rows: transform only
    prm = B.fit(tr)
    X = B.transform(te, prm, impute=True)
    _check_output(X, len(te), prm)
    assert X[B.NUM_FEATURES].notna().all().all()
