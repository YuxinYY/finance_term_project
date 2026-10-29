"""Create a cleaned, model-ready CSV and a machine-readable data summary."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from finance_project.common import project_path, write_json


def clean_dataframe(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str], int]:
    """Remove accidental CSV index columns, duplicate rows, and infinite values."""
    frame = frame.copy()
    frame.columns = [str(column).lstrip("\ufeff").strip() for column in frame.columns]
    unnamed_columns = [
        column for column in frame.columns if not column or column.lower().startswith("unnamed")
    ]
    frame = frame.drop(columns=unnamed_columns, errors="ignore")
    duplicate_count = int(frame.duplicated().sum())
    frame = frame.drop_duplicates().replace([float("inf"), float("-inf")], pd.NA)
    return frame, unnamed_columns, duplicate_count


def build_summary(
    frame: pd.DataFrame,
    source: Path,
    dropped_columns: list[str],
    duplicate_count: int,
) -> dict:
    missing = frame.isna().sum().sort_values(ascending=False)
    return {
        "source": str(source),
        "rows": int(len(frame)),
        "columns": int(frame.shape[1]),
        "duplicate_rows_removed": duplicate_count,
        "dropped_index_columns": dropped_columns,
        "missing_cells": int(frame.isna().sum().sum()),
        "top_missing_columns": [
            {"column": str(column), "missing": int(count), "missing_pct": round(count / len(frame) * 100, 2)}
            for column, count in missing.head(10).items()
            if count > 0
        ],
        "dtypes": {str(column): str(dtype) for column, dtype in frame.dtypes.items()},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/raw/train_demo_subset.csv", help="Raw CSV input path.")
    parser.add_argument("--output", help="Cleaned CSV output path. Defaults to data/processed/<input>_clean.csv.")
    parser.add_argument(
        "--summary-output",
        default="reports/metrics/data_summary.json",
        help="JSON data-quality/EDA summary path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = project_path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input data not found: {input_path}")

    output_path = project_path(args.output) if args.output else project_path(
        f"data/processed/{input_path.stem}_clean.csv"
    )
    frame = pd.read_csv(input_path)
    clean, dropped_columns, duplicate_count = clean_dataframe(frame)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(output_path, index=False)

    summary = build_summary(clean, input_path, dropped_columns, duplicate_count)
    summary["cleaned_output"] = str(output_path)
    summary_path = write_json(args.summary_output, summary)
    print(f"Wrote cleaned data: {output_path}")
    print(f"Wrote data summary: {summary_path}")


if __name__ == "__main__":
    main()
