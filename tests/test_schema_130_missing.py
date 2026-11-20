"""Pin the schema 1.3.0 missing-value decisions (docs/project-update/features/02-missing-value.md,
Desmond's commit 6bf455a) to the code that preprocess.py actually runs (D-157).

1. Method I: roe = profit / eqty_tot * 100 for every row (raw roe ignored), zero equity counts as 1;
   roe_negative = 1 iff profit < 0 and eqty_tot < 0.
2. Method G is gone: liab_lt / liab_lt_emp / debt_lt use method B even when the firm's previous
   year is 0, so a row's filled values never depend on another row.
3. Method B: the 3 industry groups with the fewest training firms fall back to the overall median.
"""

import numpy as np
import pandas as pd

from finance_project.data_processing import preprocess as pp
from test_data_processing import _raw_row

sector_medians = pp._missing["sector_medians"]


def test_method_i_recomputes_roe_for_every_row_and_flags_double_negatives():
    rows = [
        _raw_row(1, 2008),  # profit 10, eqty 50 -> 20
        {**_raw_row(2, 2008), "roe": np.nan, "profit": -10, "eqty_tot": -40},  # blank raw roe, both negative
        {**_raw_row(3, 2008), "roe": 999.0, "profit": 7, "eqty_tot": 0},  # wrong raw roe, zero equity
        {**_raw_row(4, 2008), "profit": -5},  # loss with positive equity
    ]
    frame = pd.DataFrame(rows)
    out = pp.fill_missing(frame, sector_medians(frame))
    np.testing.assert_allclose(out["roe"], [20.0, 25.0, 700.0, -10.0])
    assert out["roe_negative"].tolist() == [0, 1, 0, 0]
    assert "roe" not in pp.IMPUTE_COLS
    assert pp.FEATURE_SCHEMA_VERSION == "1.3.0"
    assert "roe_negative" in pp.FEATURE_SCHEMA["period_binary_features"]


def test_long_term_liabilities_use_method_b_not_previous_year_zero():
    train = pd.DataFrame(
        [{**_raw_row(i, 2008), "liab_lt": 20, "liab_lt_emp": 10, "debt_lt": 30} for i in range(10, 20)]
    )
    medians = sector_medians(train)
    history = pd.DataFrame(
        [
            {**_raw_row(1, 2007), "liab_lt": 0, "liab_lt_emp": 0, "debt_lt": 0},
            {**_raw_row(1, 2008), "liab_lt": np.nan, "liab_lt_emp": np.nan, "debt_lt": np.nan},
        ]
    )
    together = pp.fill_missing(history, medians).set_index("fs_year")
    alone = pp.fill_missing(history.iloc[[1]], medians).set_index("fs_year")
    # Old method G would have set 0 from the 2007 zero; method B gives median ratio * asst_tot (100).
    assert together.loc[2008, ["liab_lt", "liab_lt_emp", "debt_lt"]].tolist() == [20.0, 10.0, 30.0]
    pd.testing.assert_series_equal(together.loc[2008], alone.loc[2008])


def test_three_smallest_industry_groups_fall_back_to_overall_median():
    # Firms per group: C (sector 10) 5, G (46) 4, A (1) 1, F (41) 2, I (55) 3 -> drop A, F, I.
    spec = {10: (5, 0.10), 46: (4, 0.20), 1: (1, 0.90), 41: (2, 0.80), 55: (3, 0.70)}
    rows, firm = [], 0
    for sector, (n_firms, ratio) in spec.items():
        for _ in range(n_firms):
            firm += 1
            rows.append({**_raw_row(firm, 2008, sector=sector), "cash_and_equiv": ratio * 100})
    train = pd.DataFrame(rows)
    medians = sector_medians(train)
    assert sorted(medians.index) == ["ALL", "C", "G"]
    overall = float(train["cash_and_equiv"].div(train["asst_tot"]).median())
    holdout = pd.DataFrame(
        [{**_raw_row(900 + k, 2009, sector=s), "cash_and_equiv": np.nan} for k, s in enumerate([10, 46, 1, 41, 55])]
    )
    filled = pp.fill_missing(holdout, medians)["cash_and_equiv"].tolist()
    np.testing.assert_allclose(filled, [10.0, 20.0, overall * 100, overall * 100, overall * 100])


def test_transform_row_does_not_depend_on_other_rows_in_the_batch():
    config = pp.PreprocessingConfig(feature_mode="firm_year")
    train = pd.DataFrame([_raw_row(i, 2007, label=int(i % 4 == 0), sector=10 if i % 2 else 46) for i in range(1, 21)])
    _, artifacts = pp.fit_preprocessor(train, train, config)
    batch = pd.DataFrame(
        [
            {**_raw_row(50, 2008), "liab_lt": 0, "debt_lt": 0},
            {**_raw_row(50, 2009), "liab_lt": np.nan, "debt_lt": np.nan, "roe": np.nan},
            {**_raw_row(51, 2009, sector=46), "AR": np.nan},
        ]
    )
    full = pp.transform_with_preprocessor(batch, batch, artifacts)
    for position in range(len(batch)):
        single = batch.iloc[[position]]
        one = pp.transform_with_preprocessor(single, single, artifacts)
        key = (full["id"] == one["id"].iloc[0]) & (full["fs_year"] == one["fs_year"].iloc[0])
        pd.testing.assert_frame_equal(full.loc[key].reset_index(drop=True), one.reset_index(drop=True))
