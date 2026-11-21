"""Training-only Weight of Evidence encoding and logistic-regression scorecards."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score


@dataclass(frozen=True)
class WOELogisticConfig:
    """Configuration for a compact, regularised WoE scorecard."""

    n_bins: int = 10
    smoothing: float = 0.5
    c_values: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0)
    max_iter: int = 1_000
    random_state: int = 42


class WOEEncoder:
    """Quantile-bin numeric features and replace bins with smoothed WoE values.

    The encoder is deliberately fitted on the training partition only.  Missing
    values receive a distinct, smoothed WoE value rather than being median-filled.
    """

    def __init__(self, n_bins: int = 10, smoothing: float = 0.5) -> None:
        if n_bins < 2:
            raise ValueError("n_bins must be at least 2.")
        if smoothing <= 0:
            raise ValueError("smoothing must be positive.")
        self.n_bins = n_bins
        self.smoothing = smoothing

    def fit(self, features: pd.DataFrame, target: pd.Series) -> "WOEEncoder":
        target = pd.Series(target, index=features.index).astype(int)
        if set(target.dropna().unique()) - {0, 1}:
            raise ValueError("WoE encoding requires a binary 0/1 target.")
        if target.nunique() != 2:
            raise ValueError("WoE encoding requires both target classes in training data.")

        self.feature_columns_ = list(features.columns)
        self.edges_: dict[str, np.ndarray] = {}
        self.woe_: dict[str, np.ndarray] = {}
        self.missing_woe_: dict[str, float] = {}

        total_bad = float(target.sum())
        total_good = float((1 - target).sum())

        for column in self.feature_columns_:
            values = pd.to_numeric(features[column], errors="coerce").to_numpy(dtype=float)
            observed = np.isfinite(values)
            observed_values = values[observed]

            if observed_values.size == 0 or np.unique(observed_values).size <= 1:
                edges = np.array([-np.inf, np.inf])
            else:
                quantiles = np.linspace(0, 1, self.n_bins + 1)
                raw_edges = np.quantile(observed_values, quantiles)
                interior = np.unique(raw_edges[1:-1])
                edges = np.concatenate(([-np.inf], interior, [np.inf]))

            bin_count = len(edges) - 1
            bin_index = np.searchsorted(edges[1:-1], values[observed], side="right")
            observed_target = target.to_numpy()[observed]
            bad = np.bincount(bin_index, weights=observed_target, minlength=bin_count)
            count = np.bincount(bin_index, minlength=bin_count)
            good = count - bad

            # The extra category is the missing bin.  It is included in the
            # smoothing denominator so all transformed values are finite.
            category_count = bin_count + 1
            bad_share = (bad + self.smoothing) / (total_bad + self.smoothing * category_count)
            good_share = (good + self.smoothing) / (total_good + self.smoothing * category_count)
            self.edges_[column] = edges
            self.woe_[column] = np.log(bad_share / good_share)

            missing_target = target.to_numpy()[~observed]
            missing_bad = float(missing_target.sum())
            missing_good = float(len(missing_target) - missing_bad)
            self.missing_woe_[column] = float(
                np.log(
                    ((missing_bad + self.smoothing) / (total_bad + self.smoothing * category_count))
                    / ((missing_good + self.smoothing) / (total_good + self.smoothing * category_count))
                )
            )
        return self

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "feature_columns_"):
            raise RuntimeError("Fit the WOEEncoder before calling transform.")
        missing_columns = sorted(set(self.feature_columns_) - set(features.columns))
        if missing_columns:
            raise ValueError(f"Features are missing fitted columns: {missing_columns}")

        transformed = pd.DataFrame(index=features.index)
        for column in self.feature_columns_:
            values = pd.to_numeric(features[column], errors="coerce").to_numpy(dtype=float)
            observed = np.isfinite(values)
            output = np.full(len(values), self.missing_woe_[column], dtype=float)
            output[observed] = self.woe_[column][
                np.searchsorted(self.edges_[column][1:-1], values[observed], side="right")
            ]
            transformed[f"woe_{column}"] = output
        return transformed


@dataclass
class WOELogisticModel:
    encoder: WOEEncoder
    estimator: LogisticRegression
    selected_c: float
    feature_columns: list[str]

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        aligned = features.reindex(columns=self.feature_columns)
        encoded = self.encoder.transform(aligned)
        return self.estimator.predict_proba(encoded)[:, 1]


def fit_woe_logistic(
    train_features: pd.DataFrame,
    train_target: pd.Series,
    validation_features: pd.DataFrame,
    validation_target: pd.Series,
    config: WOELogisticConfig | None = None,
) -> tuple[WOELogisticModel, dict[str, float]]:
    """Fit the encoder/model on training rows and select regularisation on validation."""
    config = config or WOELogisticConfig()
    encoder = WOEEncoder(n_bins=config.n_bins, smoothing=config.smoothing).fit(train_features, train_target)
    train_encoded = encoder.transform(train_features)
    validation_encoded = encoder.transform(validation_features)

    candidates: list[tuple[float, float, float, LogisticRegression]] = []
    for c_value in config.c_values:
        estimator = LogisticRegression(
            C=c_value,
            solver="lbfgs",
            max_iter=config.max_iter,
            random_state=config.random_state,
        )
        estimator.fit(train_encoded, train_target)
        probabilities = estimator.predict_proba(validation_encoded)[:, 1]
        auc = float(roc_auc_score(validation_target, probabilities))
        loss = float(log_loss(validation_target, probabilities, labels=[0, 1]))
        candidates.append((auc, loss, c_value, estimator))

    # Prefer AUC, then lower log loss, then stronger regularisation.
    auc, loss, selected_c, estimator = sorted(candidates, key=lambda row: (-row[0], row[1], row[2]))[0]
    model = WOELogisticModel(
        encoder=encoder,
        estimator=estimator,
        selected_c=selected_c,
        feature_columns=list(train_features.columns),
    )
    return model, {
        "validation_roc_auc": auc,
        "validation_log_loss": loss,
        "selected_c": selected_c,
        "woe_bins_requested": float(config.n_bins),
    }
