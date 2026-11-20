"""Select transformed features using iterative within-group VIF on training only.

Run after preprocessing and before model training, once per temporal round.
The saved feature list is applied unchanged to validation, test and inference.
This module has no dependency on preprocessing notebooks or model estimators.
"""

from __future__ import annotations

import argparse
from fnmatch import fnmatchcase
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from finance_project.common import project_path

METADATA_COLUMNS = ("id", "fs_year", "pred_start_dt", "label")
ELIMINATION_COLUMNS = ("group", "iteration", "feature", "vif", "reason")
FINAL_COLUMNS = ("group", "feature", "vif", "diagnostic_rows")


def calculate_vif(correlation: pd.DataFrame) -> pd.Series:
    """Calculate individual VIFs, including when some predictors are dependent.

    Regress each standardized feature on the others using the correlation
    matrix. A singular matrix does not give unrelated predictors infinite VIF.
    An intercept is implicit through centering. Residual variance <= 1e-12 is
    treated as exact dependence. Values are never regularized to hide collinearity.
    """
    matrix = correlation.to_numpy(dtype=float)
    if not np.isfinite(matrix).all():
        raise ValueError("VIF requires a finite correlation matrix without constant columns.")
    values = []
    for position in range(len(matrix)):
        others = np.arange(len(matrix)) != position
        if not others.any():
            values.append(1.0)
            continue
        cross = matrix[others, position]
        coefficients = np.linalg.lstsq(matrix[np.ix_(others, others)], cross, rcond=None)[0]
        residual_variance = 1.0 - float(cross @ coefficients)
        values.append(np.inf if residual_variance <= 1e-12 else max(1.0, 1.0 / residual_variance))
    return pd.Series(values, index=correlation.index, dtype=float, name="vif")


def select_features(
    train: pd.DataFrame,
    feature_groups: dict[str, list[str]],
    *,
    threshold: float = 10.0,
    metadata_columns: tuple[str, ...] = METADATA_COLUMNS,
    max_rows: int = 0,
    random_state: int = 42,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Fit selection on training only; return artifact, elimination log and final VIFs.

    All training rows are used by default (max_rows=0). Optional subsampling
    uses one reproducible sample for every group and elimination iteration.
    Nonfinite values are filled with training medians for diagnostics only;
    input features and row counts are not changed by this operation.
    """
    if not np.isfinite(threshold) or threshold <= 1:
        raise ValueError("threshold must be finite and greater than 1.")
    if max_rows < 0 or max_rows == 1:
        raise ValueError("max_rows must be zero (all rows) or at least 2.")
    if len(train) < 2:
        raise ValueError("VIF selection requires at least two training rows.")
    if train.columns.duplicated().any():
        raise ValueError("Duplicate training columns are not allowed.")
    if not feature_groups:
        raise ValueError("At least one feature group must be configured.")
    features = [c for c in train.columns if c not in metadata_columns]
    if not features:
        raise ValueError("No model features in training data.")
    numeric = train[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    reasons = {}
    for column in features:
        if numeric[column].notna().sum() == 0:
            reasons[column] = "all_missing_or_nonfinite"
        elif numeric[column].nunique(dropna=True) < 2:
            reasons[column] = "constant"
    medians = numeric.median()
    if max_rows and len(numeric) > max_rows:
        numeric = numeric.sample(n=max_rows, random_state=random_state)
    numeric = numeric.fillna(medians)
    for column in features:
        if column not in reasons and numeric[column].nunique() < 2:
            reasons[column] = "constant_in_diagnostic_sample"
    active = [c for c in features if c not in reasons]
    groups = {
        group: [c for c in features if any(fnmatchcase(c, pattern) for pattern in patterns)]
        for group, patterns in feature_groups.items()
    }
    removed = set(reasons)
    log = [{"group": "precheck", "iteration": 0, "feature": c, "vif": np.nan, "reason": reason}
           for c, reason in reasons.items()]
    # Scale before centering to avoid overflow for large financial amounts.
    usable = numeric[active]
    scale = usable.abs().max().replace(0, 1)
    correlation = usable.div(scale).corr()
    iterations = {}
    for group, columns in groups.items():
        iteration = 0
        while True:
            remaining = [c for c in columns if c not in removed]
            if not remaining:
                break
            vif = calculate_vif(correlation.loc[remaining, remaining])
            highest = float(vif.max())
            if highest < threshold:
                break
            # Schema/group order has no effect on ties: use a stable name order.
            candidate = sorted(vif.index[vif == highest])[0]
            iteration += 1
            removed.add(candidate)
            log.append({"group": group, "iteration": iteration, "feature": candidate,
                        "vif": highest, "reason": "vif_at_or_above_threshold"})
        iterations[group] = iteration
    selected = [c for c in features if c not in removed]
    if not selected:
        raise ValueError("No model features survive VIF selection.")
    final = []
    for group, columns in groups.items():
        remaining = [c for c in columns if c in selected]
        vif = calculate_vif(correlation.loc[remaining, remaining])
        if not (vif < threshold).all():
            raise AssertionError(f"VIF selection did not converge for {group}.")
        final.extend({"group": group, "feature": c, "vif": float(v),
                      "diagnostic_rows": len(numeric)} for c, v in vif.items())
    grouped = {c for columns in groups.values() for c in columns}
    artifact = {
        "selection_version": 1,
        "method": "iterative_within_group_vif",
        "threshold": threshold,
        "metadata_columns": list(metadata_columns),
        "input_features": features,
        "selected_features": selected,
        "dropped_features": [c for c in features if c in removed],
        "ungrouped_features_retained": [c for c in selected if c not in grouped],
        "feature_groups": feature_groups,
        "group_iterations": iterations,
        "training_rows": len(train),
        "diagnostic_rows": len(numeric),
        "max_rows": max_rows,
        "random_state": random_state,
        "diagnostic_missing_policy": "training median fill; does not modify output values",
        "tie_break": "alphabetical feature name",
    }
    return (artifact, pd.DataFrame(log, columns=ELIMINATION_COLUMNS),
            pd.DataFrame(final, columns=FINAL_COLUMNS))


def apply_selection(frame: pd.DataFrame, artifact: dict) -> pd.DataFrame:
    """Apply a saved training feature list, preserving metadata and row order."""
    selected = artifact["selected_features"]
    missing = sorted(set(selected).difference(frame.columns))
    if missing:
        raise ValueError(f"Split lacks selected training features: {missing}")
    metadata = [c for c in artifact["metadata_columns"] if c in frame]
    return frame[[*metadata, *selected]].copy()


def run_selection(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    schema_path: str | Path = "config/feature_schema.json",
    threshold: float = 10.0,
    metadata_columns: tuple[str, ...] = METADATA_COLUMNS,
    max_rows: int = 0,
    random_state: int = 42,
) -> dict:
    """Fit once on features_train.csv, then write selected copies of all splits."""
    input_dir, output_dir = project_path(input_dir), project_path(output_dir)
    if input_dir.resolve() == output_dir.resolve():
        raise ValueError("Use a separate output directory to preserve the full transformed features.")
    schema = json.loads(project_path(schema_path).read_text(encoding="utf-8"))
    splits = {name: pd.read_csv(input_dir / f"features_{name}.csv", low_memory=False)
              for name in ("train", "validation", "test")}
    artifact, log, final = select_features(
        splits["train"], schema["feature_groups"], threshold=threshold,
        metadata_columns=metadata_columns, max_rows=max_rows, random_state=random_state,
    )
    outputs = {name: apply_selection(frame, artifact) for name, frame in splits.items()}
    artifact["schema_version"] = schema["schema_version"]
    artifact["schema_sha256"] = hashlib.sha256(
        json.dumps(schema, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    artifact["input_dir"] = str(input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in outputs.items():
        frame.to_csv(output_dir / f"features_{name}.csv", index=False)
    (output_dir / "selected_features.json").write_text(
        json.dumps(artifact, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    log.to_csv(output_dir / "vif_elimination.csv", index=False)
    final.to_csv(output_dir / "final_vif.csv", index=False)
    print(f"Kept {len(artifact['selected_features'])} of {len(artifact['input_features'])} features; "
          f"selection fitted on {artifact['diagnostic_rows']:,} training rows.")
    print(f"Wrote selected splits and diagnostics to {output_dir}")
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, help="One round's transformed feature directory.")
    parser.add_argument("--output-dir", required=True, help="Separate directory for selected features and reports.")
    parser.add_argument("--schema", default="config/feature_schema.json")
    parser.add_argument("--threshold", type=float, default=10.0)
    parser.add_argument("--max-rows", type=int, default=0, help="Diagnostic sample size; 0 uses all training rows.")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--metadata-columns", nargs="+", default=list(METADATA_COLUMNS))
    args = parser.parse_args()
    run_selection(args.input_dir, args.output_dir, schema_path=args.schema,
                  threshold=args.threshold, metadata_columns=tuple(args.metadata_columns),
                  max_rows=args.max_rows, random_state=args.random_state)


if __name__ == "__main__":
    main()
