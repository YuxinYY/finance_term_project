"""Monotone LightGBM training for the walk-forward default experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


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
    "roa": -1,
    "op_margin": -1,
    "net_prof_margin": -1,
    "int_coverage": -1,
    "financing_expense_ratio": 1,
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


def explain_lightgbm(model: LightGBMModel, features: pd.DataFrame, target: pd.Series,
                     *, max_rows: int = 3000, n_bins: int = 10,
                     repeats: int = 3, random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute centered first-order ALE and permutation importance on a test sample.

    ALE uses raw PD, quantile bins and weighted centering. Importance is the
    decrease in raw-model ROC AUC and PR AUC; it does not refit the model.
    """
    if max_rows < 2 or n_bins < 2 or repeats < 1:
        raise ValueError("max_rows and n_bins must be at least 2; repeats must be positive.")
    x = features[model.feature_columns].reset_index(drop=True)
    y = pd.Series(np.asarray(target)).reset_index(drop=True)
    if len(x) != len(y) or len(x) == 0 or not np.isfinite(x.to_numpy(dtype=float)).all():
        raise ValueError("Explanation data must be nonempty, finite and aligned with target.")
    if not y.isin([0, 1]).all() or y.nunique() != 2:
        raise ValueError("Permutation importance requires both binary target classes.")
    if len(x) > max_rows:
        from sklearn.model_selection import train_test_split
        indices, _ = train_test_split(np.arange(len(x)), train_size=max_rows,
                                      stratify=y, random_state=random_state)
        x, y = x.iloc[indices].reset_index(drop=True), y.iloc[indices].reset_index(drop=True)
    rng = np.random.default_rng(random_state)
    baseline = model.predict_proba(x)
    base_auc, base_pr = roc_auc_score(y, baseline), average_precision_score(y, baseline)
    ale_rows, importance_rows = [], []
    for feature in model.feature_columns:
        values = x[feature].to_numpy(dtype=float)
        auc_drops, pr_drops = [], []
        shuffled = x.copy()
        for _ in range(repeats):
            shuffled[feature] = rng.permutation(values)
            prediction = model.predict_proba(shuffled)
            auc_drops.append(base_auc - roc_auc_score(y, prediction))
            pr_drops.append(base_pr - average_precision_score(y, prediction))
        importance_rows.append({"feature": feature, "baseline_roc_auc": base_auc,
                                "baseline_pr_auc": base_pr, "roc_auc_drop_mean": np.mean(auc_drops),
                                "roc_auc_drop_std": np.std(auc_drops), "pr_auc_drop_mean": np.mean(pr_drops),
                                "pr_auc_drop_std": np.std(pr_drops), "repeats": repeats, "sample_rows": len(x)})
        unique = np.unique(values)
        if len(unique) == 1:
            ale_rows.append({"feature": feature, "feature_value": unique[0], "ale": 0.0,
                             "bin_count": len(x), "kind": "constant"})
            continue
        if np.isin(unique, [0, 1]).all():
            low, high = x.copy(), x.copy()
            low[feature], high[feature] = 0, 1
            difference = float(np.mean(model.predict_proba(high) - model.predict_proba(low)))
            for value in (0, 1):
                ale_rows.append({"feature": feature, "feature_value": value,
                                 "ale": difference * (value - np.mean(values)),
                                 "bin_count": int((values == value).sum()), "kind": "binary"})
            continue
        edges = np.unique(np.quantile(values, np.linspace(0, 1, n_bins + 1)))
        bins = np.clip(np.searchsorted(edges, values, side="left") - 1, 0, len(edges) - 2)
        lower, upper = x.copy(), x.copy()
        lower[feature], upper[feature] = edges[bins], edges[bins + 1]
        differences = model.predict_proba(upper) - model.predict_proba(lower)
        local, counts = [], []
        for idx in range(len(edges) - 1):
            mask = bins == idx
            counts.append(int(mask.sum()))
            local.append(float(differences[mask].mean()) if mask.any() else 0.0)
        cumulative = np.r_[0.0, np.cumsum(local)]
        center = float(np.average((cumulative[:-1] + cumulative[1:]) / 2, weights=counts))
        for idx, edge in enumerate(edges):
            ale_rows.append({"feature": feature, "feature_value": edge, "ale": cumulative[idx] - center,
                             "bin_count": counts[idx - 1] if idx else 0, "kind": "continuous"})
    return pd.DataFrame(ale_rows), pd.DataFrame(importance_rows).sort_values("roc_auc_drop_mean", ascending=False)


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
