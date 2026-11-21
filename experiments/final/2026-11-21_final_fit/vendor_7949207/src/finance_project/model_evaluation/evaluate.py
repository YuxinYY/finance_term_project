"""Evaluate the persisted baseline on its held-out split and write JSON metrics."""

from __future__ import annotations

import argparse

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, precision_score, r2_score, recall_score

from finance_project.common import project_path, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/processed/train_demo_subset_clean.csv")
    parser.add_argument("--target", required=True, help="Must match the target used in training.")
    parser.add_argument("--model", default="models/baseline.joblib")
    parser.add_argument("--output", default="reports/metrics/model_evaluation.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_bundle = joblib.load(project_path(args.model))
    if model_bundle["target"] != args.target:
        raise ValueError(
            f"Model was trained for '{model_bundle['target']}', not requested target '{args.target}'."
        )
    frame = pd.read_csv(project_path(args.data)).dropna(subset=[args.target])
    test_index = model_bundle["test_index"]
    if not set(test_index).issubset(frame.index):
        raise ValueError("The evaluation data does not match the data used to create the saved split.")
    features = frame.loc[test_index, model_bundle["feature_columns"]]
    actual = frame.loc[test_index, args.target]
    predicted = model_bundle["pipeline"].predict(features)
    task = model_bundle["task"]
    if task == "classification":
        metrics = {
            "accuracy": round(float(accuracy_score(actual, predicted)), 6),
            "precision_macro": round(float(precision_score(actual, predicted, average="macro", zero_division=0)), 6),
            "recall_macro": round(float(recall_score(actual, predicted, average="macro", zero_division=0)), 6),
            "f1_macro": round(float(f1_score(actual, predicted, average="macro", zero_division=0)), 6),
        }
    else:
        metrics = {
            "mae": round(float(mean_absolute_error(actual, predicted)), 6),
            "rmse": round(float(mean_squared_error(actual, predicted) ** 0.5), 6),
            "r2": round(float(r2_score(actual, predicted)), 6),
        }
    payload = {
        "target": args.target,
        "task": task,
        "model": type(model_bundle["pipeline"].named_steps["model"]).__name__,
        "evaluation_rows": int(len(actual)),
        "metrics": metrics,
    }
    output = write_json(args.output, payload)
    print(f"Wrote held-out evaluation metrics: {output}")


if __name__ == "__main__":
    main()
