"""Sanity tests for the shared code on synthetic data. Run: python -m pytest src/common -q"""
import numpy as np
import pandas as pd

from src.common.labels import make_labels, label_sensitivity
from src.common.metrics import (auc, accuracy_ratio, log_loss, brier, calibration_in_the_large,
                                calibration_slope_intercept, decile_table, score, paired_bootstrap)


def _synthetic(n_firms=1500, seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    for f in range(n_firms):
        risk = rng.normal()
        for yr in range(2007, 2012):
            x = risk + rng.normal(scale=0.5)
            p = 1 / (1 + np.exp(-(-3.5 + 1.2 * x)))
            rows.append((f, yr, x, rng.random() < p, p))
    return pd.DataFrame(rows, columns=["id", "fs_year", "x", "y", "p_true"])


def test_labels_windows():
    df = pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "fs_year": [2009, 2009, 2009, 2011, 2012],
        "def_date": ["2010-09-01", "2010-03-01", None, None, None],
    })
    lab = make_labels(df, stmt_col=None, start_months=6)
    # firm1: window 2010-07-01..2011-07-01 -> default inside
    assert lab.loc[0, "y"] == 1 and lab.loc[0, "label_status"] == "ok"
    # firm2: defaulted before window -> excluded
    assert lab.loc[1, "label_status"] == "already_defaulted"
    assert lab.loc[2, "y"] == 0
    # fs_year 2011: window ends 2013-07-01 > 2012-12-31 -> censored
    assert lab.loc[3, "label_status"] == "censored"
    assert lab.loc[4, "label_status"] == "censored"
    s = label_sensitivity(df, stmt_col=None)
    assert list(s["start_months"]) == [0, 3, 6]


def test_metrics_basic():
    df = _synthetic()
    y, p = df["y"].values.astype(int), df["p_true"].values
    a = auc(y, p)
    assert 0.7 < a < 0.95
    assert abs(accuracy_ratio(y, p) - (2 * a - 1)) < 1e-12
    # true probabilities should be roughly calibrated
    c = calibration_slope_intercept(y, p)
    assert abs(c["cal_slope"] - 1) < 0.2 and abs(c["cal_intercept"]) < 0.2
    # doubling all PDs -> worse log loss, intercept should go negative
    assert log_loss(y, np.clip(2 * p, 0, 0.99)) > log_loss(y, p)
    assert calibration_slope_intercept(y, np.clip(2 * p, 0, 0.99))["cal_intercept"] < -0.3
    assert brier(y, p) < brier(y, np.full_like(p, 0.5))
    cil = calibration_in_the_large(y, p)
    assert abs(cil["mean_pd"] - cil["observed_dr"]) < 0.01
    dt = decile_table(y, p)
    assert len(dt) == 10
    assert dt["observed_dr"].iloc[-1] > 3 * dt["observed_dr"].iloc[0]
    sc = score(y, p, group=df["fs_year"].values)
    assert set(sc["group"]) >= {"ALL", 2007, 2011}


def test_paired_bootstrap_detects_better_model():
    df = _synthetic()
    rng = np.random.default_rng(3)
    y = df["y"].values.astype(int)
    good = df["p_true"].values
    noisy_logit = np.log(good / (1 - good)) + rng.normal(scale=1.5, size=len(good))
    bad = 1 / (1 + np.exp(-noisy_logit))
    res = paired_bootstrap(y, good, bad, cluster=df["id"].values, n_boot=200).set_index("metric")
    assert res.loc["d_auc", "ci_low"] > 0          # good model has higher AUC
    assert res.loc["d_log_loss", "ci_high"] < 0    # and lower log loss
