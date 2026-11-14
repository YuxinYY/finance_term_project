"""Tests for leakage-safe walk-forward model development utilities."""

import numpy as np
import pandas as pd
import pytest

from finance_project.model_development.ebm_model import EBMConfig
from finance_project.model_development.lightgbm_model import (
    LightGBMConfig,
    fit_final_monotone_lightgbm,
)
from finance_project.model_development.run_walk_forward_models import (
    RunnerConfig,
    load_feature_splits,
    run_model_for_round,
)
from finance_project.model_development.woe_logistic import WOELogisticConfig, fit_woe_logistic
from finance_project.model_development.train_production_best_algo import (
    load_out_of_time_calibration_predictions,
)
from finance_project.model_development.interpret_production_lightgbm import build_feature_curve


def _split_frame(start, rows, *, omit_legal_srs=False):
    index = np.arange(start, start + rows)
    frame = pd.DataFrame(
        {
            "id": index + 1,
            "fs_year": 2007,
            "pred_start_dt": "2008-06-01",
            "label": (index % 3 == 0).astype(int),
            "cash_ratio": (index % 8) / 10,
            "debt_to_ebitda": 1 + (index % 5),
            "loss_flag": (index % 4 == 0).astype(int),
            "legal_SRL": 1,
            "legal_SRS": 0,
        }
    )
    if omit_legal_srs:
        frame = frame.drop(columns="legal_SRS")
    return frame


def _write_round(tmp_path):
    round_dir = tmp_path / "round_01"
    round_dir.mkdir()
    _split_frame(1, 30).to_csv(round_dir / "features_train.csv", index=False)
    _split_frame(101, 15, omit_legal_srs=True).to_csv(round_dir / "features_validation.csv", index=False)
    _split_frame(201, 15).to_csv(round_dir / "features_test.csv", index=False)
    return round_dir


def test_woe_logistic_keeps_missing_values_finite_and_selects_on_validation():
    train = _split_frame(1, 30)[["cash_ratio", "debt_to_ebitda", "loss_flag", "legal_SRL", "legal_SRS"]]
    validation = _split_frame(101, 15)[train.columns]
    train.loc[0, "cash_ratio"] = np.nan
    model, details = fit_woe_logistic(
        train,
        _split_frame(1, 30)["label"],
        validation,
        _split_frame(101, 15)["label"],
        WOELogisticConfig(n_bins=4, c_values=(0.1, 1.0)),
    )

    probabilities = model.predict_proba(validation)
    assert details["selected_c"] in {0.1, 1.0}
    assert np.isfinite(probabilities).all()
    assert ((probabilities > 0) & (probabilities < 1)).all()


def test_runner_aligns_absent_dummy_and_writes_woe_artifacts(tmp_path):
    round_dir = _write_round(tmp_path)
    train, validation, test, feature_columns = load_feature_splits(round_dir)
    assert feature_columns[-1] == "legal_SRS"
    assert validation["legal_SRS"].eq(0).all()
    assert len(train) == 30 and len(test) == 15

    payload = run_model_for_round(
        round_dir=round_dir,
        model_family="woe_logistic",
        config=RunnerConfig(
            output_dir=str(tmp_path / "artifacts"), run_name="test_run", figure_dir=str(tmp_path / "figures")
        ),
        woe_config=WOELogisticConfig(n_bins=4, c_values=(0.1, 1.0)),
    )

    assert payload["test"]["calibrated"]["roc_auc"] is not None
    assert (tmp_path / "artifacts" / "test_run" / "round_01" / "woe_logistic" / "model.joblib").is_file()
    assert (
        tmp_path / "artifacts" / "test_run" / "round_01" / "woe_logistic" / "test_predictions.csv"
    ).is_file()
    assert (tmp_path / "figures" / "test_run" / "round_01" / "woe_logistic" / "roc_curve.png").is_file()
    assert (tmp_path / "figures" / "test_run" / "round_01" / "woe_logistic" / "pr_curve.png").is_file()


def test_lightgbm_runner_uses_validation_for_early_stopping(tmp_path):
    pytest.importorskip("lightgbm")
    round_dir = _write_round(tmp_path)
    payload = run_model_for_round(
        round_dir=round_dir,
        model_family="lightgbm",
        config=RunnerConfig(
            output_dir=str(tmp_path / "artifacts"), run_name="test_run", figure_dir=str(tmp_path / "figures")
        ),
        lightgbm_config=LightGBMConfig(
            n_estimators=30,
            early_stopping_rounds=5,
            min_child_samples=2,
            num_leaves=4,
            n_jobs=1,
        ),
    )

    assert payload["model_details"]["best_iteration"] <= 30
    assert "cash_ratio" in payload["model_details"]["monotone_constraints"]
    assert payload["test"]["calibrated"]["pr_auc"] is not None


def test_final_lightgbm_fit_uses_fixed_iteration_count(tmp_path):
    pytest.importorskip("lightgbm")
    frame = _split_frame(1, 30)
    features = frame[["cash_ratio", "debt_to_ebitda", "loss_flag", "legal_SRL", "legal_SRS"]]
    model, details = fit_final_monotone_lightgbm(
        features,
        frame["label"],
        LightGBMConfig(n_estimators=12, min_child_samples=2, num_leaves=4, n_jobs=1),
    )

    assert details["fit_mode"] == "final_fixed_iteration"
    assert details["n_estimators"] == 12
    assert len(model.predict_proba(features)) == len(frame)


def test_production_calibration_loader_pools_out_of_time_lightgbm_predictions(tmp_path):
    for round_name, rows in {
        "round_01": pd.DataFrame({"id": [1, 2], "label": [0, 1], "pd_raw": [0.1, 0.8]}),
        "round_02": pd.DataFrame({"id": [3, 4], "label": [0, 1], "pd_raw": [0.2, 0.7]}),
    }.items():
        destination = tmp_path / round_name / "lightgbm"
        destination.mkdir(parents=True)
        rows.to_csv(destination / "test_predictions.csv", index=False)

    pooled = load_out_of_time_calibration_predictions(tmp_path)

    assert len(pooled) == 4
    assert set(pooled["source_round"]) == {"round_01", "round_02"}
    assert pooled["label"].tolist() == [0, 1, 0, 1]


def test_feature_curve_uses_full_population_bins_and_sample_shap_values():
    values = pd.Series(np.arange(1, 101, dtype=float))
    labels = pd.Series((values > 70).astype(int))
    sample_index = np.arange(0, 100, 2)
    curve, curve_type = build_feature_curve(
        values,
        labels,
        values.iloc[sample_index],
        pd.Series(np.linspace(-0.2, 0.3, len(sample_index)), index=sample_index),
        n_bins=5,
    )

    assert curve_type == "quantile"
    assert curve["n_obs"].sum() == 100
    assert curve["n_defaults"].sum() == 30
    assert curve["shap_sample_rows"].sum() == 50
    assert curve["smoothed_default_rate"].between(0, 1).all()


def test_ebm_runner_writes_global_importance_and_shape_figures(tmp_path):
    pytest.importorskip("interpret")
    round_dir = _write_round(tmp_path)
    payload = run_model_for_round(
        round_dir=round_dir,
        model_family="ebm",
        config=RunnerConfig(
            output_dir=str(tmp_path / "artifacts"), run_name="test_run", figure_dir=str(tmp_path / "figures")
        ),
        ebm_config=EBMConfig(
            outer_bags=1,
            max_rounds=50,
            early_stopping_rounds=5,
            min_samples_leaf=2,
            n_jobs=1,
        ),
    )

    assert payload["model_details"]["feature_set"] == "core"
    importance_path = payload["artifacts"]["global_feature_importance"]
    assert pd.read_csv(importance_path)["feature"].tolist()
    assert payload["artifacts"]["feature_curve_figures"]
