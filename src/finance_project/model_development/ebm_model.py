"""Explainable Boosting Machine training and explanation exports."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from finance_project.model_development.lightgbm_model import MONOTONE_DIRECTIONS


# A compact set of financial ratios plus the flags/context needed to interpret
# exceptional statements.  The ``full`` option uses every transformed feature.
CORE_EBM_FEATURES = (
    "size",
    "st_debt_share",
    "lt_debt_to_capital",
    "bank_debt_conc",
    "cash_ratio",
    "cash_to_financing_exp_ratio",
    "cur_asst_ratio",
    "AR_ratio",
    "change_in_ar_turnover",
    "payables_ratio",
    "wc_ratio",
    "cur_debt_ratio",
    "quick_ratio",
    "asset_turnover",
    "annualized_change_in_roa",
    "ebitda_ratio",
    "op_margin",
    "net_prof_margin",
    "int_coverage",
    "cash_to_debt",
    "ebitda_to_debt",
    "accruals",
    "neg_bs_flag",
    "debt_detail_missing",
    "loss_flag",
    "history_records",
    "no_prior_observation",
    "ateco_sector_te",
)


@dataclass(frozen=True)
class EBMConfig:
    feature_set: str = "core"
    max_bins: int = 64
    interactions: int = 0
    validation_size: float = 0.15
    outer_bags: int = 4
    learning_rate: float = 0.03
    max_rounds: int = 5_000
    early_stopping_rounds: int = 100
    min_samples_leaf: int = 50
    reg_lambda: float = 1.0
    random_state: int = 42
    n_jobs: int = -1


@dataclass
class EBMModel:
    estimator: Any
    feature_columns: list[str]
    constraints: dict[str, int]
    config: EBMConfig

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        return self.estimator.predict_proba(features.reindex(columns=self.feature_columns))[:, 1]


def select_ebm_features(features: pd.DataFrame, feature_set: str) -> list[str]:
    """Return a documented core or full EBM feature set."""
    if feature_set == "full":
        return list(features.columns)
    if feature_set != "core":
        raise ValueError("feature_set must be 'core' or 'full'.")
    selected = [column for column in CORE_EBM_FEATURES if column in features.columns]
    selected.extend(column for column in features.columns if column.startswith("legal_"))
    selected.extend(column for column in features.columns if column.startswith("city_"))
    if not selected:
        raise ValueError("None of the core EBM features were found.")
    return selected


def fit_ebm(
    train_features: pd.DataFrame,
    train_target: pd.Series,
    config: EBMConfig | None = None,
) -> tuple[EBMModel, dict[str, Any]]:
    """Fit an EBM using an internal training-only validation split for stopping."""
    try:
        from interpret.glassbox import ExplainableBoostingClassifier
    except ImportError as error:  # pragma: no cover - depends on active environment
        raise ImportError(
            "The EBM model requires 'interpret'. Install the project dependencies "
            "or activate the project's daily environment."
        ) from error

    config = config or EBMConfig()
    requested_features = select_ebm_features(train_features, config.feature_set)
    dropped_all_missing = [
        column for column in requested_features if not train_features[column].notna().any()
    ]
    feature_columns = [column for column in requested_features if column not in dropped_all_missing]
    if not feature_columns:
        raise ValueError("All selected EBM features are missing in the training partition.")
    constraints = {column: MONOTONE_DIRECTIONS.get(column, 0) for column in feature_columns}
    estimator = ExplainableBoostingClassifier(
        feature_names=feature_columns,
        max_bins=config.max_bins,
        interactions=config.interactions,
        validation_size=config.validation_size,
        outer_bags=config.outer_bags,
        learning_rate=config.learning_rate,
        max_rounds=config.max_rounds,
        early_stopping_rounds=config.early_stopping_rounds,
        min_samples_leaf=config.min_samples_leaf,
        reg_lambda=config.reg_lambda,
        missing="separate",
        monotone_constraints=[constraints[column] for column in feature_columns],
        random_state=config.random_state,
        n_jobs=config.n_jobs,
    )
    estimator.fit(train_features.reindex(columns=feature_columns), train_target)
    model = EBMModel(
        estimator=estimator,
        feature_columns=feature_columns,
        constraints=constraints,
        config=config,
    )
    return model, {
        "feature_set": config.feature_set,
        "dropped_all_missing_features": dropped_all_missing,
        "monotone_constraints": constraints,
        "config": asdict(config),
        "early_stopping": "Internal validation split drawn only from the training partition.",
    }


def export_ebm_explanations(
    model: EBMModel,
    *,
    artifact_dir: str | Path,
    figure_dir: str | Path | None = None,
    top_features: int = 10,
) -> dict[str, Any]:
    """Write global importance and top univariate EBM shape curves."""
    global_explanation = model.estimator.explain_global()
    global_data = global_explanation.data()
    importance = pd.DataFrame(
        {"feature": global_data["names"], "importance": global_data["scores"]}
    ).sort_values("importance", ascending=False, ignore_index=True)
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    importance_path = artifact_dir / "global_feature_importance.csv"
    importance.to_csv(importance_path, index=False)

    paths: dict[str, Any] = {"global_feature_importance": str(importance_path)}
    if figure_dir is None:
        return paths

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    curve_dir = Path(figure_dir) / "ebm_feature_curves"
    curve_dir.mkdir(parents=True, exist_ok=True)
    curve_paths = []
    term_names = list(global_data["names"])
    for rank, feature in enumerate(importance["feature"].head(top_features), start=1):
        feature_index = term_names.index(feature)
        term_data = global_explanation.data(feature_index)
        # With interactions disabled, global terms are univariate and can be
        # displayed as log-odds contributions across the raw feature range.
        y_values = np.asarray(term_data["scores"], dtype=float)
        raw_x_values = np.asarray(term_data["names"])
        try:
            x_values = raw_x_values.astype(float)
        except (TypeError, ValueError):
            x_values = np.arange(len(raw_x_values), dtype=float)
        # EBM reports continuous-term bin edges (n + 1 values) alongside n
        # scores.  Plot a score at its bin midpoint.  Categorical terms use
        # one label/value per score and fall through unchanged.
        if len(x_values) == len(y_values) + 1:
            x_values = (x_values[:-1] + x_values[1:]) / 2
        elif len(x_values) != len(y_values):
            x_values = np.arange(len(y_values), dtype=float)
        figure, axis = plt.subplots(figsize=(6, 4))
        axis.plot(x_values, y_values, color="#009E73", linewidth=2)
        axis.axhline(0, color="0.45", linewidth=1)
        axis.set(
            xlabel=str(feature),
            ylabel="EBM contribution to log-odds",
            title=f"EBM shape: {feature}",
        )
        axis.grid(alpha=0.2)
        figure.tight_layout()
        safe_feature = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(feature))
        curve_path = curve_dir / f"{rank:02d}_{safe_feature}.png"
        figure.savefig(curve_path, dpi=180)
        plt.close(figure)
        curve_paths.append(str(curve_path))
    paths["feature_curve_figures"] = curve_paths
    return paths
