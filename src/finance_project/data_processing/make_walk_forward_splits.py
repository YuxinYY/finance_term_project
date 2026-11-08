"""Create leakage-safe, expanding-window walk-forward dataset splits.

The input must already contain a 12-month default label for each usable
firm-year. Each ``(id, fs_year)`` pair is a distinct prediction sample: a
FY2008 statement predicts default from June 2009 through May 2010. For each
round, preceding fiscal years form an expanding training window. Validation is
either a stratified, ID-disjoint sample of that window (``company_random``) or
the next complete fiscal year (``temporal_year``). In temporal validation, a
firm can appear in earlier training years and later validation/test years.
Each round also includes ``history.csv`` through the test fiscal year, so
preprocessing can construct features without accessing future statements.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from math import ceil
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from finance_project.common import project_path, write_json


@dataclass(frozen=True)
class WalkForwardSplit:
    """One expanding-window train/validation/test split."""

    round_number: int
    train_years: tuple[int, ...]
    validation_year: int | None
    test_year: int
    validation_strategy: str
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    history: pd.DataFrame


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


def _validation_ids(
    historical: pd.DataFrame,
    *,
    id_column: str,
    target_column: str,
    validation_size: float,
    random_state: int,
) -> set:
    """Select validation companies, stratifying by historical default status when possible."""
    company_target = historical.groupby(id_column, sort=False)[target_column].max()
    if len(company_target) < 2:
        raise ValueError("Each historical window needs at least two distinct company IDs.")

    # Stratifying company IDs preserves firms with a prior positive label in both
    # partitions. Fall back to an unstratified deterministic sample only when a
    # class is too small for sklearn's stratified split.
    class_counts = company_target.value_counts(dropna=False)
    validation_rows = max(1, ceil(len(company_target) * validation_size))
    can_stratify = (
        company_target.nunique(dropna=False) > 1
        and class_counts.min() >= 2
        and validation_rows >= len(class_counts)
        and len(company_target) - validation_rows >= len(class_counts)
    )
    _, validation_ids = train_test_split(
        company_target.index,
        test_size=validation_size,
        random_state=random_state,
        stratify=company_target if can_stratify else None,
    )
    return set(validation_ids)


def make_walk_forward_splits(
    frame: pd.DataFrame,
    *,
    id_column: str = "id",
    year_column: str = "fs_year",
    target_column: str = "label",
    first_train_year: int | None = None,
    rounds: int = 4,
    validation_strategy: str = "company_random",
    validation_size: float = 0.2,
    random_state: int = 42,
) -> list[WalkForwardSplit]:
    """Return expanding-window splits using the requested validation strategy.

    With fiscal years 2007--2011, ``first_train_year=2007`` and ``rounds=4``,
    the output is:

    * round 1: train/validation FY2007, test FY2008;
    * round 2: train/validation FY2007--2008, test FY2009;
    * round 3: train/validation FY2007--2009, test FY2010;
    * round 4: train/validation FY2007--2010, test FY2011.

    ``company_random`` preserves the original ID-disjoint validation behaviour.
    ``temporal_year`` uses the whole next fiscal year for validation and the
    following year for test. With FY2007 as the first training year and three
    rounds: train FY2007 / validation FY2008 / test FY2009; then train
    FY2007--2008 / validation FY2009 / test FY2010; then train FY2007--2009 /
    validation FY2010 / test FY2011. ``history`` contains records from the
    first training year through the test fiscal year.
    """
    if rounds < 1:
        raise ValueError("rounds must be at least 1.")
    if validation_strategy not in {"company_random", "temporal_year"}:
        raise ValueError("validation_strategy must be 'company_random' or 'temporal_year'.")
    if validation_strategy == "company_random" and not 0 < validation_size < 1:
        raise ValueError("validation_size must be strictly between 0 and 1.")

    required_columns = {id_column, year_column, target_column}
    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        raise ValueError(f"Input is missing required columns: {sorted(missing_columns)}")
    if frame[[id_column, year_column, target_column]].isna().any().any():
        raise ValueError(
            f"'{id_column}', '{year_column}', and '{target_column}' must be non-missing before splitting."
        )

    data = frame.copy()
    data[year_column] = pd.to_numeric(data[year_column], errors="raise").astype(int)
    start_year = int(data[year_column].min()) if first_train_year is None else first_train_year
    first_test_year = start_year + (2 if validation_strategy == "temporal_year" else 1)
    test_years = list(range(first_test_year, first_test_year + rounds))
    available_years = set(data[year_column].unique())
    unavailable_years = [year for year in test_years if year not in available_years]
    if unavailable_years:
        raise ValueError(
            "The requested walk-forward rounds require test fiscal years that are absent from the input: "
            f"{unavailable_years}."
        )

    splits: list[WalkForwardSplit] = []
    for round_number, test_year in enumerate(test_years, start=1):
        test = data.loc[data[year_column] == test_year].copy()
        history = data.loc[
            (data[year_column] >= start_year) & (data[year_column] <= test_year)
        ].copy()
        validation_year: int | None = None
        if validation_strategy == "temporal_year":
            validation_year = test_year - 1
            train = data.loc[
                (data[year_column] >= start_year) & (data[year_column] < validation_year)
            ].copy()
            validation = data.loc[data[year_column] == validation_year].copy()
            validation_ids: set = set(validation[id_column].unique())
        else:
            historical = data.loc[
                (data[year_column] >= start_year) & (data[year_column] < test_year)
            ].copy()
            if historical.empty:
                raise ValueError(f"Round {round_number} has an empty historical window.")
            validation_ids = _validation_ids(
                historical,
                id_column=id_column,
                target_column=target_column,
                validation_size=validation_size,
                random_state=random_state + round_number - 1,
            )
            validation = historical.loc[historical[id_column].isin(validation_ids)].copy()
            train = historical.loc[~historical[id_column].isin(validation_ids)].copy()

        train_ids = set(train[id_column].unique())
        if train.empty or validation.empty:
            raise ValueError(f"Round {round_number} has an empty train or validation set.")
        if test.empty:
            raise ValueError(f"Round {round_number} has an empty test set.")
        if validation_strategy == "company_random" and train_ids.intersection(validation_ids):
            raise AssertionError("Train and validation company IDs must be disjoint.")

        splits.append(
            WalkForwardSplit(
                round_number=round_number,
                train_years=tuple(int(year) for year in sorted(train[year_column].unique())),
                validation_year=validation_year,
                test_year=test_year,
                validation_strategy=validation_strategy,
                train=train,
                validation=validation,
                test=test,
                history=history,
            )
        )
    return splits


def _split_summary(split: WalkForwardSplit, *, id_column: str, target_column: str) -> dict:
    train_ids = set(split.train[id_column].unique())
    validation_ids = set(split.validation[id_column].unique())
    return {
        "round": split.round_number,
        "validation_strategy": split.validation_strategy,
        "train_years": list(split.train_years),
        "validation_year": split.validation_year,
        "test_year": split.test_year,
        "train_rows": int(len(split.train)),
        "validation_rows": int(len(split.validation)),
        "test_rows": int(len(split.test)),
        "history_rows": int(len(split.history)),
        "train_companies": int(len(train_ids)),
        "validation_companies": int(len(validation_ids)),
        "test_companies": int(split.test[id_column].nunique()),
        "train_validation_id_overlap": int(len(train_ids.intersection(validation_ids))),
        "train_default_rate": round(float(split.train[target_column].mean()), 6),
        "validation_default_rate": round(float(split.validation[target_column].mean()), 6),
        "test_default_rate": round(float(split.test[target_column].mean()), 6),
    }


def write_walk_forward_splits(
    splits: list[WalkForwardSplit],
    output_dir: Path,
    *,
    id_column: str,
    target_column: str,
) -> list[dict]:
    """Write one directory per round and return manifest entries."""
    manifest_rounds = []
    for split in splits:
        round_dir = output_dir / f"round_{split.round_number:02d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        split.train.to_csv(round_dir / "train.csv", index=False)
        split.validation.to_csv(round_dir / "validation.csv", index=False)
        split.test.to_csv(round_dir / "test.csv", index=False)
        split.history.to_csv(round_dir / "history.csv", index=False)

        summary = _split_summary(split, id_column=id_column, target_column=target_column)
        summary["files"] = {
            "train": str(round_dir / "train.csv"),
            "validation": str(round_dir / "validation.csv"),
            "test": str(round_dir / "test.csv"),
            "history": str(round_dir / "history.csv"),
        }
        manifest_rounds.append(summary)
    return manifest_rounds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="data/processed/labeled_data.csv",
        help="Labeled firm-year CSV input path.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/walk_forward_temporal",
        help="Directory in which to write round_XX/train.csv, validation.csv, test.csv, and history.csv.",
    )
    parser.add_argument("--id-column", default="id")
    parser.add_argument("--year-column", default="fs_year")
    parser.add_argument("--target-column", default="label")
    parser.add_argument("--first-train-year", type=int, default=None)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument(
        "--validation-strategy",
        choices=["temporal_year", "company_random"],
        default="temporal_year",
        help=(
            "temporal_year uses the complete fiscal year before test; company_random holds out "
            "a stratified, ID-disjoint sample of the historical window."
        ),
    )
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--summary-output",
        default="reports/metrics/walk_forward_splits.json",
        help="JSON split-manifest path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = project_path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input data not found: {input_path}")

    raw = pd.read_csv(input_path)
    data, dropped_columns, duplicate_count = clean_dataframe(raw)
    splits = make_walk_forward_splits(
        data,
        id_column=args.id_column,
        year_column=args.year_column,
        target_column=args.target_column,
        first_train_year=args.first_train_year,
        rounds=args.rounds,
        validation_strategy=args.validation_strategy,
        validation_size=args.validation_size,
        random_state=args.random_state,
    )

    output_dir = project_path(args.output_dir)
    manifest_rounds = write_walk_forward_splits(
        splits,
        output_dir,
        id_column=args.id_column,
        target_column=args.target_column,
    )
    summary_path = write_json(
        args.summary_output,
        {
            "source": str(input_path),
            "output_dir": str(output_dir),
            "rows_after_cleaning": int(len(data)),
            "duplicate_rows_removed": duplicate_count,
            "dropped_index_columns": dropped_columns,
            "id_column": args.id_column,
            "year_column": args.year_column,
            "target_column": args.target_column,
            "validation_strategy": args.validation_strategy,
            "validation_size": args.validation_size,
            "random_state": args.random_state,
            "rounds": manifest_rounds,
        },
    )
    print(f"Wrote {len(splits)} walk-forward rounds to: {output_dir}")
    print(f"Wrote split manifest: {summary_path}")


if __name__ == "__main__":
    main()
