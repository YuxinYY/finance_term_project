"""Bootstrap confidence intervals and paired model comparisons for walk-forward test predictions.

Models are not retrained.  For each scope (all rounds pooled, then each round alone) the test
rows are resampled with replacement ``--n-bootstrap`` times, and every model is scored on the
*same* resampled rows.  This yields a percentile interval per model and, because the rows are
shared, a paired interval for each metric difference between two models.  A difference is
treated as significant when its interval excludes zero; overlapping per-model intervals are not
a valid test because the models are scored on the same data.
"""

from __future__ import annotations

import argparse
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from finance_project.common import project_path, write_json

KEY_COLUMNS = ["id", "fs_year"]
METRICS = {
    "roc_auc": roc_auc_score,
    "pr_auc": average_precision_score,
}


def _calibration_in_the_large(actual: np.ndarray, probabilities: np.ndarray) -> float:
    return float(probabilities.mean() - actual.mean())


METRICS["calibration_in_the_large"] = _calibration_in_the_large


def load_predictions(
    artifact_dir: str, run_name: str, rounds: list[int], models: list[str], probability_column: str
) -> pd.DataFrame:
    """Return one row per (round, id, fs_year) with the label and one probability column per model."""
    root = project_path(artifact_dir) / run_name
    frames = []
    for round_number in rounds:
        merged: pd.DataFrame | None = None
        for model in models:
            path = root / f"round_{round_number:02d}" / model / "test_predictions.csv"
            if not path.exists():
                raise FileNotFoundError(f"Test predictions not found: {path}")
            frame = pd.read_csv(path, usecols=[*KEY_COLUMNS, "label", probability_column])
            frame = frame.rename(columns={probability_column: model}).sort_values(KEY_COLUMNS, ignore_index=True)
            if merged is None:
                merged = frame
                continue
            same_rows = len(frame) == len(merged) and (frame[KEY_COLUMNS] == merged[KEY_COLUMNS]).all().all()
            if not same_rows or not (frame["label"] == merged["label"]).all():
                raise ValueError(f"Round {round_number}: {model} was not scored on the same test rows.")
            merged[model] = frame[model]
        merged.insert(0, "round", round_number)
        frames.append(merged)
    return pd.concat(frames, ignore_index=True)


def bootstrap_scope(
    frame: pd.DataFrame, models: list[str], n_bootstrap: int, alpha: float, random_state: int
) -> dict:
    """Bootstrap metrics for each model and each model pair on one set of rows."""
    actual = frame["label"].to_numpy()
    probabilities = {model: frame[model].to_numpy() for model in models}
    rng = np.random.default_rng(random_state)
    draws = {metric: {model: np.empty(n_bootstrap) for model in models} for metric in METRICS}
    n_rows = len(frame)
    for i in range(n_bootstrap):
        index = rng.integers(0, n_rows, n_rows)
        sample_actual = actual[index]
        if sample_actual.min() == sample_actual.max():  # ponytail: ~impossible at 1.5% default rate; NaN is dropped below
            for metric in METRICS:
                for model in models:
                    draws[metric][model][i] = np.nan
            continue
        for model in models:
            sample_probabilities = probabilities[model][index]
            for metric, function in METRICS.items():
                draws[metric][model][i] = function(sample_actual, sample_probabilities)

    lower, upper = 100 * alpha / 2, 100 * (1 - alpha / 2)

    def summarize(values: np.ndarray, point: float) -> dict:
        values = values[~np.isnan(values)]
        return {
            "point": float(point),
            "boot_mean": float(values.mean()),
            "boot_std": float(values.std(ddof=1)),
            "ci_lower": float(np.percentile(values, lower)),
            "ci_upper": float(np.percentile(values, upper)),
        }

    result: dict = {
        "rows": n_rows,
        "defaults": int(actual.sum()),
        "models": {},
        "differences": {},
    }
    for model in models:
        result["models"][model] = {
            metric: summarize(draws[metric][model], function(actual, probabilities[model]))
            for metric, function in METRICS.items()
        }
    for first, second in combinations(models, 2):
        pair = {}
        for metric, function in METRICS.items():
            difference = draws[metric][first] - draws[metric][second]
            point = function(actual, probabilities[first]) - function(actual, probabilities[second])
            summary = summarize(difference, point)
            summary["excludes_zero"] = bool(summary["ci_lower"] > 0 or summary["ci_upper"] < 0)
            pair[metric] = summary
        result["differences"][f"{first} - {second}"] = pair
    return result


def summary_table(results: dict) -> pd.DataFrame:
    """Flatten the per-model and per-pair results into one tidy table."""
    rows = []
    for scope, scope_result in results.items():
        for model, metrics in scope_result["models"].items():
            for metric, stats in metrics.items():
                rows.append({"scope": scope, "comparison": model, "metric": metric, **stats})
        for pair, metrics in scope_result["differences"].items():
            for metric, stats in metrics.items():
                rows.append({"scope": scope, "comparison": pair, "metric": metric, **stats})
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="model_artifacts", help="Root containing <run-name>/round_XX/<model>/.")
    parser.add_argument("--run-name", default="baseline_v1")
    parser.add_argument("--rounds", default="1,2,3", help="Comma-separated round numbers.")
    parser.add_argument("--models", default="woe_logistic,lightgbm,ebm", help="Comma-separated model families.")
    parser.add_argument(
        "--probability-column",
        choices=["pd_calibrated", "pd_raw"],
        default="pd_calibrated",
    )
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=0.05, help="1 - confidence level.")
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rounds = sorted({int(item) for item in args.rounds.split(",") if item.strip()})
    models = [item.strip() for item in args.models.split(",") if item.strip()]
    if len(models) < 1:
        raise ValueError("At least one model is required.")
    predictions = load_predictions(args.output_dir, args.run_name, rounds, models, args.probability_column)

    scopes = {"pooled": predictions}
    scopes.update({f"round_{r:02d}": predictions[predictions["round"] == r] for r in rounds})
    results = {}
    for name, frame in scopes.items():
        print(f"Bootstrapping {name} ({len(frame):,} rows, {args.n_bootstrap} replications)")
        results[name] = bootstrap_scope(frame, models, args.n_bootstrap, args.alpha, args.random_state)

    output_dir = project_path(args.output_dir) / args.run_name / "bootstrap"
    json_path = write_json(
        output_dir / "bootstrap_results.json",
        {
            "config": {**vars(args), "rounds": rounds, "models": models},
            "results": results,
        },
    )
    csv_path = output_dir / "bootstrap_summary.csv"
    summary_table(results).to_csv(csv_path, index=False)
    print(f"Wrote bootstrap results: {json_path}")
    print(f"Wrote bootstrap summary: {csv_path}")


if __name__ == "__main__":
    main()
