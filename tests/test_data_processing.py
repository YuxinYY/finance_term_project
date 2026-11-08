# Tests CSV cleanup and leakage-safe, ID-disjoint walk-forward dataset splits.

import pandas as pd
import pytest

from finance_project.data_processing.make_walk_forward_splits import (
    clean_dataframe,
    make_walk_forward_splits,
    write_walk_forward_splits,
)
from finance_project.data_processing.preprocess import (
    PreprocessingConfig,
    bootstrap_minority_class,
    calculate_vif,
    feature_group_collinearity,
    feature_schema_coverage,
    fit_preprocessor,
    transform_with_preprocessor,
)


def test_clean_dataframe_removes_unnamed_index_columns_and_duplicates():
    source = pd.DataFrame({"Unnamed: 0": [0, 1, 1], "profit": [5, 6, 6]})
    cleaned, dropped, duplicates = clean_dataframe(source)

    assert list(cleaned.columns) == ["profit"]
    assert dropped == ["Unnamed: 0"]
    assert duplicates == 1


def test_walk_forward_splits_expand_by_year_and_keep_validation_ids_out_of_train(tmp_path):
    source = pd.DataFrame(
        [
            {"id": company_id, "fs_year": year, "label": int(company_id % 2 == 0)}
            for year in range(2007, 2012)
            for company_id in range(1, 21)
        ]
    )

    splits = make_walk_forward_splits(
        source,
        first_train_year=2007,
        rounds=4,
        validation_size=0.2,
        random_state=7,
    )

    assert [split.test_year for split in splits] == [2008, 2009, 2010, 2011]
    assert [split.train_years for split in splits] == [
        (2007,),
        (2007, 2008),
        (2007, 2008, 2009),
        (2007, 2008, 2009, 2010),
    ]
    for split in splits:
        train_ids = set(split.train["id"])
        validation_ids = set(split.validation["id"])
        assert train_ids.isdisjoint(validation_ids)
        assert set(split.train["fs_year"]) == set(split.train_years)
        assert set(split.validation["fs_year"]) == set(split.train_years)
        assert set(split.test["fs_year"]) == {split.test_year}
        assert set(split.history["fs_year"]) == set(split.train_years) | {split.test_year}

    manifest = write_walk_forward_splits(
        splits,
        tmp_path,
        id_column="id",
        target_column="label",
    )
    assert len(manifest) == 4
    assert all(round_info["train_validation_id_overlap"] == 0 for round_info in manifest)
    for round_number in range(1, 5):
        round_dir = tmp_path / f"round_{round_number:02d}"
        assert (round_dir / "train.csv").is_file()
        assert (round_dir / "validation.csv").is_file()
        assert (round_dir / "test.csv").is_file()
        assert (round_dir / "history.csv").is_file()


def test_walk_forward_splits_support_temporal_year_validation_with_expected_overlap():
    source = pd.DataFrame(
        [
            {"id": company_id, "fs_year": year, "label": int(company_id % 2 == 0)}
            for year in range(2007, 2012)
            for company_id in range(1, 6)
        ]
    )

    splits = make_walk_forward_splits(
        source,
        first_train_year=2007,
        rounds=3,
        validation_strategy="temporal_year",
    )

    assert [split.test_year for split in splits] == [2009, 2010, 2011]
    assert [split.validation_year for split in splits] == [2008, 2009, 2010]
    assert [split.train_years for split in splits] == [
        (2007,),
        (2007, 2008),
        (2007, 2008, 2009),
    ]
    for split in splits:
        assert set(split.validation["fs_year"]) == {split.validation_year}
        assert set(split.test["fs_year"]) == {split.test_year}
        assert set(split.history["fs_year"]) == set(split.train_years) | {
            split.validation_year,
            split.test_year,
        }
        assert set(split.train["id"]) == set(split.validation["id"])


def _raw_row(company_id, year, label=None, *, sector=10, legal="SRL", current_assets=20, ar=10):
    row = {
        "id": company_id,
        "fs_year": year,
        "HQ_city": 1,
        "legal_struct": legal,
        "ateco_sector": sector,
        "asst_intang_fixed": 5,
        "asst_tang_fixed": 10,
        "asst_fixed_fin": 0,
        "asst_current": current_assets,
        "AR": ar,
        "cash_and_equiv": 10,
        "asst_tot": 100,
        "goodwill": 0,
        "eqty_tot": 50,
        "liab_lt": 0,
        "liab_lt_emp": 0,
        "debt_bank_st": 5,
        "debt_bank_lt": 10,
        "debt_fin_st": 0,
        "debt_fin_lt": 0,
        "AP_st": 5,
        "AP_lt": 0,
        "debt_st": 10,
        "debt_lt": 10,
        "rev_operating": 50,
        "COGS": 30,
        "prof_operations": 20,
        "inc_financing": 0,
        "exp_financing": 2,
        "prof_financing": 0,
        "inc_extraord": 0,
        "taxes": 5,
        "profit": 10,
        "ebitda": 25,
        "cf_operations": 15,
        "wc_net": 10,
        "margin_fin": 35,
        "roa": 20,
        "pred_start_dt": f"{year + 1}-06-01",
    }
    if label is not None:
        row["label"] = label
    return row


def test_preprocessor_fits_on_train_and_uses_decay_weighted_history_for_transform():
    train = pd.DataFrame(
        [
            _raw_row(1, 2007, 0, sector=10),
            _raw_row(2, 2007, 1, sector=10),
            _raw_row(3, 2007, 0, sector=20),
            _raw_row(4, 2007, 1, sector=20),
        ]
    )
    test = pd.DataFrame([_raw_row(5, 2008, 0, sector=99, legal="UNSEEN", current_assets=30)])
    history = pd.concat(
        [
            train.drop(columns="label"),
            pd.DataFrame([_raw_row(5, 2007, sector=99, legal="UNSEEN", current_assets=10, ar=-1)]),
            test.drop(columns="label"),
        ],
        ignore_index=True,
    )
    config = PreprocessingConfig(target_encoding_folds=2, random_state=7)

    transformed_train, artifacts = fit_preprocessor(train, history, config)
    transformed_test = transform_with_preprocessor(test, history, artifacts)

    assert len(transformed_train) == len(train)
    assert len(transformed_test) == 1
    assert "past_num_defaults" not in transformed_train.columns
    assert "ateco_sector" not in transformed_train.columns
    assert "legal_struct" not in transformed_train.columns
    assert transformed_test.loc[0, "history_records"] == 2
    assert transformed_test.loc[0, "neg_bs_flag"] == 1
    assert transformed_test.loc[0, "ateco_sector_te"] == artifacts.industry_global_mean
    assert transformed_test.loc[0, "cur_asst_ratio"] == pytest.approx(7 / 30)
    # SRL is the fitted reference category, so it is intentionally omitted.
    # Unseen legal structures are represented by zero across fitted dummies.
    assert "legal_SRL" not in transformed_test.columns
    assert artifacts.legal_reference_category == "SRL"
    assert artifacts.city_reference_category == "1"


def test_preprocessor_implements_feature_schema_and_group_collinearity_diagnostics():
    train = pd.DataFrame(
        [
            _raw_row(1, 2007, 0, sector=10, legal="SRL"),
            _raw_row(2, 2007, 1, sector=10, legal="SPA"),
            _raw_row(3, 2007, 0, sector=20, legal="SRL"),
            _raw_row(4, 2007, 1, sector=20, legal="SPA"),
        ]
    )
    train.loc[1, "HQ_city"] = 2
    train.loc[3, "HQ_city"] = 2
    transformed, artifacts = fit_preprocessor(
        train,
        train.drop(columns="label"),
        PreprocessingConfig(target_encoding_folds=2, random_state=7),
    )

    coverage = feature_schema_coverage(artifacts.feature_columns)
    assert coverage.loc[~coverage["schema_feature"].str.endswith("*"), "implemented"].all()
    assert coverage.loc[coverage["schema_feature"].eq("legal_*"), "implemented"].item()
    assert coverage.loc[coverage["schema_feature"].eq("city_*"), "implemented"].item()

    diagnostics = feature_group_collinearity(transformed, artifacts.feature_columns)
    assert set(diagnostics) == {
        "business_structural",
        "leverage_capital_structure",
        "liquidity_activity",
        "profitability_growth",
        "coverage_cash_flow",
    }
    assert {"feature", "vif", "status", "sample_rows"}.issubset(
        diagnostics["business_structural"]["vif"].columns
    )


def test_calculate_vif_marks_perfectly_collinear_features():
    features = pd.DataFrame({"x": [1, 2, 3, 4], "twice_x": [2, 4, 6, 8], "constant": 1})

    vif, covariance, correlation = calculate_vif(features, list(features))

    assert set(vif.loc[vif["status"] == "singular_matrix", "feature"]) == {"x", "twice_x"}
    assert vif.loc[vif["feature"] == "constant", "status"].item() == "constant"
    assert covariance.index.tolist() == ["x", "twice_x"]
    assert correlation.columns.tolist() == ["x", "twice_x"]


def test_firm_year_feature_mode_uses_direct_anchor_values_not_decay_aggregation():
    train = pd.DataFrame(
        [
            _raw_row(1, 2007, 0, sector=10),
            _raw_row(2, 2007, 1, sector=10),
            _raw_row(3, 2007, 0, sector=20),
            _raw_row(4, 2007, 1, sector=20),
        ]
    )
    test = pd.DataFrame([_raw_row(5, 2008, 0, sector=10, current_assets=30, ar=15)])
    history = pd.concat(
        [
            train.drop(columns="label"),
            pd.DataFrame([_raw_row(5, 2007, sector=10, current_assets=10, ar=-1)]),
            test.drop(columns="label"),
        ],
        ignore_index=True,
    )
    config = PreprocessingConfig(feature_mode="firm_year", target_encoding_folds=2, random_state=7)

    _, artifacts = fit_preprocessor(train, history, config)
    transformed_test = transform_with_preprocessor(test, history, artifacts)

    assert "history_records" not in transformed_test.columns
    assert "no_prior_observation" not in transformed_test.columns
    assert transformed_test.loc[0, "cur_asst_ratio"] == pytest.approx(30 / 100)
    assert transformed_test.loc[0, "neg_bs_flag"] == 0


def test_bootstrap_minority_class_targets_requested_ratio_without_changing_features():
    source = pd.DataFrame(
        {
            "id": range(12),
            "feature": range(12),
            "label": [0] * 10 + [1] * 2,
        }
    )

    resampled, summary = bootstrap_minority_class(
        source,
        minority_to_majority_ratio=0.2,
        random_state=7,
    )

    assert summary["applied"] is False
    assert resampled["label"].value_counts().to_dict() == {0: 10, 1: 2}
    # A 1:5 target with ten majority rows already needs only two minority rows.
    # Raising the target to 1:2 forces replacement sampling.
    resampled, summary = bootstrap_minority_class(
        source,
        minority_to_majority_ratio=0.5,
        random_state=7,
    )
    assert summary["minority_rows_after"] == 5
    assert resampled["label"].value_counts().to_dict() == {0: 10, 1: 5}
    assert set(resampled.columns) == set(source.columns)
