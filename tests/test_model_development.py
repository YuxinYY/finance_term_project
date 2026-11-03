"""Tests for leakage-safe walk-forward model development utilities."""

import numpy as np
import pandas as pd
import pytest

from finance_project.model_development.lightgbm_model import LightGBMConfig
from finance_project.model_development.run_walk_forward_models import (
    RunnerConfig,
    load_feature_splits,
    run_model_for_round,
)
from finance_project.model_development.woe_logistic import WOELogisticConfig, fit_woe_logistic


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
        config=RunnerConfig(model_dir=str(tmp_path / "models"), report_dir=str(tmp_path / "reports")),
        woe_config=WOELogisticConfig(n_bins=4, c_values=(0.1, 1.0)),
    )

    assert payload["test"]["calibrated"]["roc_auc"] is not None
    assert (tmp_path / "models" / "round_01" / "woe_logistic.joblib").is_file()
    assert (tmp_path / "reports" / "round_01" / "woe_logistic_test_predictions.csv").is_file()


def test_lightgbm_runner_uses_validation_for_early_stopping(tmp_path):
    pytest.importorskip("lightgbm")
    round_dir = _write_round(tmp_path)
    payload = run_model_for_round(
        round_dir=round_dir,
        model_family="lightgbm",
        config=RunnerConfig(model_dir=str(tmp_path / "models"), report_dir=str(tmp_path / "reports")),
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
