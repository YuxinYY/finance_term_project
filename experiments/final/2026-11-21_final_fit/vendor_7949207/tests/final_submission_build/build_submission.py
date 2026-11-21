"""Developer build: export compatible fitted artifacts and create a standalone ZIP.

No model or statistics are fitted here. The two input artifacts must already
have been fitted with the current training pipeline and schema.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
RUNTIME_FILES = ["harness.py", "inference.py", "missing_rules.py", "anomaly_rules.py",
                 "feature_rules.py", "bundle_paths.py", "README.md", "requirements.txt",
                 "rules_provenance.json", "config/feature_schema.json",
                 "data/interim/ateco_sector_english.csv"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preprocessor", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-zip", type=Path, default=ROOT / "turquoise.zip")
    args = parser.parse_args()
    sys.path.insert(0, str(PROJECT / "src"))
    import joblib
    from finance_project.data_processing.preprocess import (
        PreprocessingArtifacts, PreprocessingConfig, FEATURE_SCHEMA_HASH,
    )
    from finance_project.model_development.run_walk_forward_models import (
        ProbabilityCalibrator, RunnerConfig,
    )
    # Artifacts created by python -m can refer to CLI classes under __main__.
    for cls in (PreprocessingArtifacts, PreprocessingConfig, ProbabilityCalibrator, RunnerConfig):
        setattr(sys.modules["__main__"], cls.__name__, cls)
    preprocessor = joblib.load(args.preprocessor)
    model_bundle = joblib.load(args.model)
    if preprocessor.feature_schema_hash != FEATURE_SCHEMA_HASH:
        raise ValueError(
            f"Preprocessor schema {preprocessor.feature_schema_version} is stale. "
            "Refit preprocessing AND the model with the current schema before packaging."
        )
    schema = json.loads((ROOT / "config/feature_schema.json").read_text())
    canonical = json.dumps(schema, sort_keys=True, separators=(",", ":"))
    if hashlib.sha256(canonical.encode()).hexdigest() != FEATURE_SCHEMA_HASH:
        raise ValueError("Standalone rules are stale. Run sync_rules.py first.")
    provenance = json.loads((ROOT / "rules_provenance.json").read_text())
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    for relative, expected in provenance.items():
        if sha(PROJECT / relative) != expected:
            raise ValueError(f"Source changed: {relative}. Run sync_rules.py first.")
    model = model_bundle["model"]
    if preprocessor.config.feature_mode != "firm_year":
        raise ValueError("This harness supports firm_year artifacts only.")
    if model.feature_columns != preprocessor.feature_columns:
        raise ValueError("Model and preprocessor feature orders differ.")
    if "roe_negative" not in model.feature_columns:
        raise ValueError("Model lacks the current schema's roe_negative feature.")
    calibrator = model_bundle["calibrator"]
    if calibrator.method != "platt":
        raise ValueError("This exporter requires a fitted Platt calibrator.")
    estimator = model.estimator
    iteration = int(estimator.best_iteration_ or estimator.booster_.num_trees())
    parameters = {
        "config": asdict(preprocessor.config),
        "schema_version": preprocessor.feature_schema_version,
        "schema_hash": preprocessor.feature_schema_hash,
        "feature_columns": preprocessor.feature_columns,
        "model_feature_names": estimator.booster_.feature_name(),
        "medians": json.loads(preprocessor.sector_medians.to_json(orient="split", double_precision=15)),
        "industry_mapping": preprocessor.industry_mapping.to_dict(),
        "industry_global_mean": float(preprocessor.industry_global_mean),
        "fallback_pd": float(preprocessor.industry_global_mean),
        "fallback_pd_source": "fitted training-anchor default rate",
        "legal_categories": preprocessor.legal_categories,
        "city_categories": preprocessor.city_categories,
        "best_iteration": iteration,
        "calibration": {"method": "platt", "epsilon": 1e-6,
                        "coefficient": float(calibrator.estimator.coef_[0, 0]),
                        "intercept": float(calibrator.estimator.intercept_[0])},
    }
    # Validate everything before writing the deployable parameter files.
    payload = json.dumps(parameters, indent=2, allow_nan=False) + "\n"
    artifacts = ROOT / "artifacts"
    artifacts.mkdir(exist_ok=True)
    estimator.booster_.save_model(str(artifacts / "lightgbm.txt"), num_iteration=iteration)
    (artifacts / "preprocessing.json").write_text(payload)
    runtime = RUNTIME_FILES + ["artifacts/lightgbm.txt", "artifacts/preprocessing.json"]
    manifest = {
        "bundle_sha256": {relative: sha(ROOT / relative) for relative in runtime},
        "source_sha256": {"preprocessor": sha(args.preprocessor), "model": sha(args.model)},
        "feature_count": len(model.feature_columns),
        "schema_version": preprocessor.feature_schema_version,
        "invalid_feature_policy": "frozen training default-rate PD for every invalid row",
    }
    (artifacts / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    args.output_zip.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(args.output_zip, "w", compression=ZIP_DEFLATED) as archive:
        for relative in runtime + ["artifacts/manifest.json"]:
            archive.write(ROOT / relative, relative)
    print(f"Built {args.output_zip} with {len(model.feature_columns)} frozen features.")


if __name__ == "__main__":
    main()
