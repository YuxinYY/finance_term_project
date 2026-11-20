"""Unit tests for the round-2 feature set `full2` (feature-architect, 2026-11-02).

Run: python -m pytest src/features/test_build_v2.py -q
Synthetic data only (re-uses the v1 fixture); no panel rows are read.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.features import build as B
from src.features.test_build import _synthetic


@pytest.fixture(scope="module")
def train():
    return _synthetic()


@pytest.fixture(scope="module")
def p2(train):
    return B.fit(train, feature_set="full2")


def _check(X, n):
    assert len(X) == n
    assert list(X.columns) == B.FULL2_FEATURES
    num = X[[c for c in B.FULL2_FEATURES if c not in B.CAT_FEATURES]].to_numpy(float)
    assert not np.isinf(num).any()


def test_v1_unchanged(train, p2):
    """Adding the v2 set must not change any v1 parameter or v1 output."""
    p1 = B.fit(train)
    p2_wo = {k: v for k, v in p2.items() if k != "v2"}
    assert json.dumps(p1, sort_keys=True) == json.dumps(p2_wo, sort_keys=True)
    X1 = B.transform(train, p1)
    X2 = B.transform(train, p2, feature_set="full2")
    pd.testing.assert_frame_equal(X1, X2[B.ALL_FEATURES])
    pd.testing.assert_frame_equal(B.transform(train, p1, impute=True),
                                  B.transform(train, p2, impute=True, feature_set="full2")[B.ALL_FEATURES])
    assert B.FULL2_FEATURES[:len(B.ALL_FEATURES)] == B.ALL_FEATURES
    assert len(set(B.FULL2_FEATURES)) == len(B.FULL2_FEATURES)


def test_json_roundtrip_and_reuse(train, p2):
    p = json.loads(json.dumps(p2))
    pd.testing.assert_frame_equal(B.transform(train, p, feature_set="full2"),
                                  B.transform(train, p2, feature_set="full2"))


def test_single_row_equals_batch(train, p2):
    X = B.transform(train, p2, impute=True, feature_set="full2")
    for i in [0, 5, 999]:
        xi = B.transform(train.iloc[[i]], p2, impute=True, feature_set="full2")
        pd.testing.assert_frame_equal(xi, X.iloc[[i]])


def test_impute_fills_everything(train, p2):
    Xi = B.transform(train, p2, impute=True, feature_set="full2")
    _check(Xi, len(train))
    assert Xi[B.NEW_FEATURES_V2].notna().all().all()
    Z = B.design_matrix(Xi, p2, B.FULL2_FEATURES)
    assert np.isfinite(Z.to_numpy(float)).all()


def test_messy_row(p2):
    messy = pd.DataFrame([{
        "asst_tot": " 2,500,000 ", "eqty_tot": "NA", "margin_fin": "100000",
        "asst_intang_fixed": "0", "asst_tang_fixed": "n.a.", "asst_fixed_fin": "", "asst_current": "#DIV/0!",
        "debt_st": "0", "debt_lt": "-", "rev_operating": "0", "AR": "5000", "COGS": "abc",
        "exp_financing": "10", "ebitda": "-10", "goodwill": "-50", "profit": "inf", "cf_operations": None,
        "HQ_city": " 999 ", "legal_struct": "XYZ", "ateco_sector": "28.0",
    }])
    X = B.transform(messy, p2, feature_set="full2")
    _check(X, 1)
    # zero revenue with positive interest -> upper winsor bound (slog scale)
    assert X.loc[0, "interest_to_sales"] == pytest.approx(B.slog(p2["v2"]["winsor"]["interest_to_sales"][1]))
    # negative personnel cost (goodwill < ebitda) -> NaN, not a number
    assert np.isnan(X.loc[0, "personnel_to_sales"])
    assert X.loc[0, "operating_margin"] != X.loc[0, "operating_margin"]   # rev <= 0 -> NaN
    # unseen province -> prior (0); ATECO 28 seen
    assert X.loc[0, "province_te"] == 0.0 and X.loc[0, "region_te"] == 0.0
    assert X.loc[0, "ateco_div_te"] == pytest.approx(p2["v2"]["te"]["ateco_div_te"]["full"]["enc"]["28"])
    assert X.loc[0, "n_core_imputed"] >= 1
    Xc = B.transform(messy, p2, feature_set="full2_candidates")
    assert Xc.loc[0, "n_missing_raw"] >= 20
    Xi = B.transform(messy, p2, impute=True, feature_set="full2")
    assert Xi[B.NEW_FEATURES_V2].notna().all().all()


def test_all_nan_row_empty_frame_and_garbage(p2):
    row = pd.DataFrame([{c: np.nan for c in B.RAW_NUM + B.RAW_CAT}])
    X = B.transform(row, p2, feature_set="full2")
    _check(X, 1)
    assert B.transform(row, p2, feature_set="full2_candidates").loc[0, "n_missing_raw"] == len(B.MISS_COUNT_FIELDS)
    assert X.loc[0, "n_core_imputed"] == len(B.CORE_NUM)
    assert X.loc[0, "ateco_div_te"] == 0.0
    Xi = B.transform(row, p2, feature_set="full2", impute=True)
    assert Xi[B.NEW_FEATURES_V2].notna().all().all()
    E = B.transform(row.iloc[:0], p2, feature_set="full2")
    assert len(E) == 0 and list(E.columns) == B.FULL2_FEATURES
    G = B.transform(pd.DataFrame({"foo": ["x", "y"]}), p2, feature_set="full2", impute=True)
    _check(G, 2)


def test_denominator_rules(p2):
    base = dict(asst_tot=2e6, eqty_tot=1.5e6, cash_and_equiv=6e5, cf_operations=1e5,
                rev_operating=1e6, prof_operations=5e4, exp_financing=1e4, AP_st=1e5,
                goodwill=3e5, ebitda=1e5, ateco_sector="41", HQ_city="15")
    rows = pd.DataFrame([
        base,                                                  # 0 net cash, positive cash flow
        {**base, "cf_operations": -1e5},                       # 1 net cash, burning cash
        {**base, "cash_and_equiv": 1e5},                       # 2 normal: 1e5 / 4e5
        {**base, "rev_operating": -5.0},                       # 3 negative revenue
        {**base, "eqty_tot": -5e5, "cash_and_equiv": 0.0},     # 4 negative equity, liabilities > assets
    ])
    X = B.transform(rows, p2, feature_set="full2_candidates")   # net_debt_coverage is a rejected candidate
    lo, hi = p2["v2"]["winsor"]["net_debt_coverage"]
    assert X.loc[0, "net_debt_coverage"] == pytest.approx(B.slog(hi))
    assert X.loc[1, "net_debt_coverage"] == pytest.approx(B.slog(np.clip(0.0, lo, hi)))
    assert X.loc[2, "net_debt_coverage"] == pytest.approx(B.slog(np.clip(0.25, lo, hi)))
    assert X.loc[4, "net_debt_coverage"] == pytest.approx(B.slog(np.clip(1e5 / 2.5e6, lo, hi)))
    for c in ["operating_margin", "interest_to_sales", "payables_to_sales", "personnel_to_sales"]:
        assert np.isnan(X.loc[3, c])
    assert X.loc[2, "personnel_to_sales"] == pytest.approx(
        B.slog(np.clip(0.2, *p2["v2"]["winsor"]["personnel_to_sales"])))


def test_sector_percentile(train, p2):
    X = B.transform(train, p2, feature_set="full2")
    for b in B.PCT_BASE:
        v = X[f"{b}_sect_pct"].dropna()
        assert v.between(0, 1).all()
        # monotone in the base feature within a sector (same grid)
        sect = B._section(B.coerce_input(train)["ateco_sector"])
        s0 = sect.iloc[0]
        m = (sect == s0) & X[b].notna()
        d = X.loc[m, [b, f"{b}_sect_pct"]].sort_values(b)
        assert (np.diff(d[f"{b}_sect_pct"].to_numpy()) >= -1e-12).all()
    # unseen sector -> pooled grid, still in [0, 1]
    one = train.iloc[[0]].copy()
    one["ateco_sector"] = "99"
    v = B.transform(one, p2, feature_set="full2")["equity_ratio_sect_pct"].iloc[0]
    assert 0 <= v <= 1


def test_target_encoding_oof_and_leakage(train, p2):
    te = p2["v2"]["te"]["ateco_div_te"]
    assert len(te["oof"]) == B.TE_FOLDS
    assert B.TE_M_BOUNDS[0] <= te["full"]["m"] <= B.TE_M_BOUNDS[1]
    Xo = B.transform(train, p2, feature_set="full2", oof=True)
    Xf = B.transform(train, p2, feature_set="full2")
    assert not np.allclose(Xo["ateco_div_te"], Xf["ateco_div_te"])   # OOF differs from in-sample
    pd.testing.assert_frame_equal(Xo.drop(columns=B.V2_TE), Xf.drop(columns=B.V2_TE))
    # firm grouping: every row of a firm gets the same fold
    ids = pd.Series([5, 5, 5, 7, 7])
    f = B._firm_fold(ids, B.TE_FOLDS)
    assert len(set(f[:3])) == 1 and len(set(f[3:])) == 1
    # a row's own label cannot move its OOF encoding: flip labels in one fold
    fold = B._firm_fold(train["id"], B.TE_FOLDS)
    t2 = train.copy()
    t2.loc[fold == 0, "y"] = 1 - t2.loc[fold == 0, "y"]
    p_alt = B.fit(t2, feature_set="full2")
    assert p_alt["v2"]["te"]["ateco_div_te"]["oof"][0] == te["oof"][0]
    with pytest.raises(ValueError):
        B.transform(train.drop(columns=["id"]), p2, feature_set="full2", oof=True)


def test_no_future_or_label_inputs(train, p2):
    """transform must not read y, def_date, fs_year, or any column other than raw inputs."""
    t = train.copy()
    t["def_date"] = "2020-01-01"
    t["y"] = 1 - t["y"]
    t["fs_year"] = 2099
    pd.testing.assert_frame_equal(B.transform(t, p2, feature_set="full2"),
                                  B.transform(train, p2, feature_set="full2"))


def test_expected_signs_full2():
    s = B.expected_signs(B.FULL2_FEATURES)
    assert set(B.NEW_FEATURES_V2) <= set(s)
    assert set(s.values()) <= {-1, 0, 1}
    assert B.expected_signs() == B.expected_signs(B.NUM_FEATURES + B.FLAG_FEATURES)


def test_rejected_candidates_not_in_set():
    assert set(B.V2_REJECTED) == {"net_debt_coverage", "n_missing_raw"}
    assert not set(B.V2_REJECTED) & set(B.FULL2_FEATURES)
    assert len(B.FULL2_FEATURES) == 45


def test_needs_v2_params(train):
    p1 = B.fit(train.iloc[:500])
    with pytest.raises(ValueError):
        B.transform(train.iloc[:5], p1, feature_set="full2")
