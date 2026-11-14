"""Monotone LightGBM training for the walk-forward default experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LightGBMConfig:
    n_estimators: int = 5_000
    learning_rate: float = 0.03
    num_leaves: int = 15
    min_child_samples: int = 200
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_lambda: float = 1.0
    early_stopping_rounds: int = 150
    random_state: int = 42
    n_jobs: int = -1


# +1 means a larger value must not decrease PD; -1 means it must not increase PD.
# Features without a defensible financial direction are deliberately unconstrained.
MONOTONE_DIRECTIONS: dict[str, int] = {
    "equity_ratio": -1,
    "st_debt_share": 1,
    "lt_debt_to_capital": 1,
    "cash_ratio": -1,
    "cash_to_financing_exp_ratio": -1,
    "AR_ratio": 1,
    "payables_ratio": 1,
    "wc_ratio": -1,
    "cur_debt_ratio": -1,
    "quick_ratio": -1,
    "annualized_change_in_roa": -1,
    "roa": -1,
    "op_margin": -1,
    "net_prof_margin": -1,
    "int_coverage": -1,
    "financing_expense_ratio": 1,
    "financing_income_ratio": -1,
    "operating_cash_flow_ratio": -1,
    "ebitda_to_debt": -1,
    "loss_flag": 1,
    "ateco_sector_te": 1,
}


@dataclass
class LightGBMModel:
    estimator: Any
    feature_columns: list[str]
    constraints: dict[str, int]
    config: LightGBMConfig

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        aligned = features.reindex(columns=self.feature_columns)
        return self.estimator.predict_proba(aligned)[:, 1]


def fit_monotone_lightgbm(
    train_features: pd.DataFrame,
    train_target: pd.Series,
    validation_features: pd.DataFrame,
    validation_target: pd.Series,
    config: LightGBMConfig | None = None,
) -> tuple[LightGBMModel, dict[str, Any]]:
    """Fit LightGBM and use the supplied validation period for early stopping."""
    try:
        from lightgbm import LGBMClassifier, early_stopping, log_evaluation
    except ImportError as error:  # pragma: no cover - depends on active environment
        raise ImportError(
            "LightGBM is required for this model. Install the project dependencies "
            "or activate the project's daily environment."
        ) from error

    config = config or LightGBMConfig()
    feature_columns = list(train_features.columns)
    constraints = {column: MONOTONE_DIRECTIONS.get(column, 0) for column in feature_columns}
    estimator = LGBMClassifier(
        objective="binary",
        n_estimators=config.n_estimators,
        learning_rate=config.learning_rate,
        num_leaves=config.num_leaves,
        min_child_samples=config.min_child_samples,
        subsample=config.subsample,
        colsample_bytree=config.colsample_bytree,
        reg_lambda=config.reg_lambda,
        monotone_constraints=[constraints[column] for column in feature_columns],
        random_state=config.random_state,
        n_jobs=config.n_jobs,
        verbosity=-1,
    )
    estimator.fit(
        train_features.reindex(columns=feature_columns),
        train_target,
        eval_set=[(validation_features.reindex(columns=feature_columns), validation_target)],
        eval_metric=["auc", "average_precision", "binary_logloss"],
        callbacks=[early_stopping(config.early_stopping_rounds, verbose=False), log_evaluation(0)],
    )
    model = LightGBMModel(
        estimator=estimator,
        feature_columns=feature_columns,
        constraints=constraints,
        config=config,
    )
    return model, {
        "best_iteration": int(estimator.best_iteration_ or config.n_estimators),
        "monotone_constraints": constraints,
        "config": asdict(config),
    }


def fit_final_monotone_lightgbm(
    train_features: pd.DataFrame,
    train_target: pd.Series,
    config: LightGBMConfig | None = None,
) -> tuple[LightGBMModel, dict[str, Any]]:
    """Fit the final fixed-iteration LightGBM model for deployment.

    The production tree count must have been selected before this step (for
    example, from walk-forward early-stopping results).  This deliberately has
    no validation input and therefore cannot use early stopping: all labeled
    observations remain available to estimate the final deployed model.
    """
    try:
        from lightgbm import LGBMClassifier
    except ImportError as error:  # pragma: no cover - depends on active environment
        raise ImportError(
            "LightGBM is required for this model. Install the project dependencies "
            "or activate the project's daily environment."
        ) from error

    config = config or LightGBMConfig()
    feature_columns = list(train_features.columns)
    constraints = {column: MONOTONE_DIRECTIONS.get(column, 0) for column in feature_columns}
    estimator = LGBMClassifier(
        objective="binary",
        n_estimators=config.n_estimators,
        learning_rate=config.learning_rate,
        num_leaves=config.num_leaves,
        min_child_samples=config.min_child_samples,
        subsample=config.subsample,
        colsample_bytree=config.colsample_bytree,
        reg_lambda=config.reg_lambda,
        monotone_constraints=[constraints[column] for column in feature_columns],
        random_state=config.random_state,
        n_jobs=config.n_jobs,
        verbosity=-1,
    )
    estimator.fit(train_features.reindex(columns=feature_columns), train_target)
    model = LightGBMModel(
        estimator=estimator,
        feature_columns=feature_columns,
        constraints=constraints,
        config=config,
    )
    return model, {
        "fit_mode": "final_fixed_iteration",
        "n_estimators": config.n_estimators,
        "monotone_constraints": constraints,
        "config": asdict(config),
    }
