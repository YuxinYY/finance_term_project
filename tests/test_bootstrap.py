"""Tests for the paired bootstrap used to compare walk-forward models."""

import numpy as np
import pandas as pd

from finance_project.model_evaluation.bootstrap import bootstrap_scope


def _scored_frame(rows=4000, seed=0):
    rng = np.random.default_rng(seed)
    label = (rng.random(rows) < 0.1).astype(int)
    signal = rng.normal(size=rows)
    return pd.DataFrame(
        {
            "label": label,
            "strong": 1 / (1 + np.exp(-(2.0 * label + signal))),
            "weak": 1 / (1 + np.exp(-(0.2 * label + signal))),
            "twin": 1 / (1 + np.exp(-(2.0 * label + signal))),
        }
    )


def test_paired_bootstrap_separates_strong_from_weak_and_not_identical_models():
    result = bootstrap_scope(_scored_frame(), ["strong", "weak", "twin"], 200, 0.05, 1)
    assert result["differences"]["strong - weak"]["roc_auc"]["excludes_zero"]
    assert not result["differences"]["strong - twin"]["roc_auc"]["excludes_zero"]
    interval = result["models"]["strong"]["roc_auc"]
    assert interval["ci_lower"] <= interval["point"] <= interval["ci_upper"]


def test_bootstrap_is_reproducible_for_a_fixed_seed():
    frame = _scored_frame()
    first = bootstrap_scope(frame, ["strong", "weak"], 50, 0.05, 7)
    second = bootstrap_scope(frame, ["strong", "weak"], 50, 0.05, 7)
    assert first == second
