"""Train a reproducible random-forest baseline for an explicitly chosen target."""

from __future__ import annotations

import argparse

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from finance_project.common import project_path, write_json


def infer_task(target: pd.Series, requested_task: str) -> str:
    if requested_task != "auto":
        return requested_task
    if not pd.api.types.is_numeric_dtype(target) or target.nunique() <= 20:
        return "classification"
    return "regression"


def build_pipeline(features: pd.DataFrame, task: str, random_state: int) -> Pipeline:
    numeric_columns = features.select_dtypes(include="number").columns.tolist()
    categorical_columns = [column for column in features.columns if column not in numeric_columns]
    transformers = []
    if numeric_columns:
        transformers.append(("numeric", SimpleImputer(strategy="median"), numeric_columns))
    if categorical_columns:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("encode", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_columns,
            )
        )
    if not transformers:
        raise ValueError("No usable feature columns remain after removing the target.")
    estimator = (
        RandomForestClassifier(n_estimators=300, random_state=random_state, n_jobs=-1)
        if task == "classification"
        else RandomForestRegressor(n_estimators=300, random_state=random_state, n_jobs=-1)
    )
    return Pipeline([("preprocess", ColumnTransformer(transformers)), ("model", estimator)])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/processed/train_demo_subset_clean.csv")
    parser.add_argument("--target", required=True, help="Approved prediction target column.")
    parser.add_argument("--task", choices=["auto", "classification", "regression"], default="auto")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--model-output", default="models/baseline.joblib")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = project_path(args.data)
    frame = pd.read_csv(data_path)
    if args.target not in frame.columns:
        raise ValueError(f"Target '{args.target}' is not in {data_path}. Available columns: {list(frame.columns)}")
    frame = frame.dropna(subset=[args.target]).copy()
    target = frame.pop(args.target)
    task = infer_task(target, args.task)
    stratify = target if task == "classification" and target.value_counts().min() >= 2 else None
    train_index, test_index = train_test_split(
        frame.index, test_size=args.test_size, random_state=args.random_state, stratify=stratify
    )
    pipeline = build_pipeline(frame, task, args.random_state)
    pipeline.fit(frame.loc[train_index], target.loc[train_index])

    model_path = project_path(args.model_output)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": pipeline,
            "target": args.target,
            "task": task,
            "feature_columns": frame.columns.tolist(),
            "test_index": test_index.tolist(),
        },
        model_path,
    )
    write_json(
        "reports/metrics/training_run.json",
        {
            "data": str(data_path),
            "target": args.target,
            "task": task,
            "model": type(pipeline.named_steps["model"]).__name__,
            "training_rows": int(len(train_index)),
            "held_out_rows": int(len(test_index)),
            "random_state": args.random_state,
            "model_path": str(model_path),
        },
    )
    print(f"Trained {task} baseline and wrote model: {model_path}")


if __name__ == "__main__":
    main()
