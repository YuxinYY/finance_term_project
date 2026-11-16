"""Developer-only export of existing round-03 artifacts; never fits a model.

Run from the repository using its daily environment and PYTHONPATH=src.
The resulting tests/harness_submission folder is independently runnable.
"""

from __future__ import annotations

import ast
import hashlib
import json
import shutil
from dataclasses import asdict
from pathlib import Path

import joblib
import lightgbm
import numpy as np
import pandas as pd
import scipy

# CLI-created artifacts refer to these names under __main__ when unpickled.
from finance_project.data_processing.preprocess import (
    PreprocessingArtifacts, PreprocessingConfig, FEATURE_SCHEMA_HASH,
)
from finance_project.model_development.run_walk_forward_models import (
    ProbabilityCalibrator, RunnerConfig,
)

ROOT = Path(__file__).resolve().parents[2]
DESTINATION = ROOT / "tests/harness_submission"


def main() -> None:
    preprocessor_path = ROOT / "models/preprocessing/temporal_firm_year_round_03.joblib"
    model_path = ROOT / "model_artifacts/temporal_firm_year_v2/round_03/lightgbm/model.joblib"
    artifacts = joblib.load(preprocessor_path)
    bundle = joblib.load(model_path)
    if artifacts.feature_schema_hash != FEATURE_SCHEMA_HASH:
        raise ValueError("Preprocessor does not match the current feature schema.")
    if artifacts.config.feature_mode != "firm_year":
        raise ValueError("Expected the round-03 firm_year preprocessor.")
    if artifacts.feature_columns != bundle["model"].feature_columns:
        raise ValueError("Preprocessing and model features differ.")
    if bundle["calibrator"].method != "platt":
        raise ValueError("Expected the fitted Platt calibration artifact.")
    artifact_dir = DESTINATION / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    sources = [preprocessor_path, model_path]
    for relative in ("config/feature_schema.json", "data/interim/ateco_sector_english.csv"):
        target = DESTINATION / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
        sources.append(ROOT / relative)
    for notebook, filename in (("01_flag_anomalies", "anomaly_rules.py"),
                               ("02_missing_values", "missing_rules.py")):
        path = ROOT / f"notebooks/features/{notebook}.ipynb"
        cells = json.loads(path.read_text())["cells"]
        source = "\n\n".join("".join(c["source"]) for c in cells
                               if "export" in c.get("metadata", {}).get("tags", []))
        (DESTINATION / filename).write_text(
            "# Frozen notebook export; no fitting is performed by the harness.\n"
            "import numpy as np\nimport pandas as pd\nfrom bundle_paths import project_path\n\n"
            + source + "\n"
        )
        sources.append(path)
    preprocessing_source = ROOT / "src/finance_project/data_processing/preprocess.py"
    source = preprocessing_source.read_text()
    selected = {"_REGION_PROVINCES", "REGION_BY_PROVINCE", "_region", "_safe_divide",
                "make_period_features", "_period_numeric_features", "_period_binary_features",
                "_validate_period_feature_contract"}
    extracted = []
    for node in ast.parse(source).body:
        names = {node.name} if isinstance(node, ast.FunctionDef) else set()
        if isinstance(node, ast.Assign):
            names = {t.id for t in node.targets if isinstance(t, ast.Name)}
        if names & selected:
            extracted.append(ast.get_source_segment(source, node))
    if len(extracted) != len(selected):
        raise ValueError("Could not extract every required frozen feature rule.")
    (DESTINATION / "feature_rules.py").write_text(
        '"""Frozen feature rules from preprocess.py; inference uses no labels."""\n'
        "from __future__ import annotations\nimport json\nimport numpy as np\nimport pandas as pd\n"
        "from bundle_paths import project_path\n"
        'FEATURE_SCHEMA = json.loads(project_path("config/feature_schema.json").read_text())\n\n'
        + "\n\n".join(extracted) + "\n"
    )
    sources.append(preprocessing_source)
    estimator = bundle["model"].estimator
    best_iteration = int(estimator.best_iteration_ or estimator.booster_.num_trees())
    estimator.booster_.save_model(str(artifact_dir / "lightgbm.txt"), num_iteration=best_iteration)
    calibration = bundle["calibrator"].estimator
    # pandas handles NaN as JSON null, rather than non-standard JSON NaN tokens.
    medians = json.loads(artifacts.sector_medians.to_json(orient="split", double_precision=15))
    parameters = {
        "config": asdict(artifacts.config),
        "schema_version": artifacts.feature_schema_version,
        "schema_hash": artifacts.feature_schema_hash,
        "feature_columns": artifacts.feature_columns,
        "model_feature_names": estimator.booster_.feature_name(),
        "medians": medians,
        "industry_mapping": artifacts.industry_mapping.to_dict(),
        "industry_global_mean": artifacts.industry_global_mean,
        # This is the training-anchor default rate stored during target encoding.
        # Freeze it at export; the harness never derives it from testing data.
        "fallback_pd": float(artifacts.industry_global_mean),
        "fallback_pd_source": "round-03 training-anchor default rate",
        "legal_categories": artifacts.legal_categories,
        "city_categories": artifacts.city_categories,
        "best_iteration": best_iteration,
        "calibration": {"method": "platt", "epsilon": 1e-6,
                        "coefficient": float(calibration.coef_[0, 0]),
                        "intercept": float(calibration.intercept_[0])},
    }
    (artifact_dir / "preprocessing.json").write_text(json.dumps(parameters, indent=2, allow_nan=False) + "\n")
    files = [p for p in DESTINATION.rglob("*") if p.is_file()
             and "__pycache__" not in p.parts and p.name != "manifest.json"]
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    manifest = {
        "source_run": "temporal_firm_year_v2/round_03/lightgbm",
        "feature_count": len(artifacts.feature_columns),
        "best_iteration": best_iteration,
        "versions": {m.__name__: m.__version__ for m in (np, pd, scipy, lightgbm)},
        "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in sources},
        "bundle_sha256": {str(p.relative_to(DESTINATION)): sha(p) for p in files},
        "invalid_feature_policy": "frozen training default-rate PD; preserve every input row in order",
        "fallback_pd": parameters["fallback_pd"],
    }
    (artifact_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Exported {len(artifacts.feature_columns)} features and {best_iteration} trees to {DESTINATION}.")


if __name__ == "__main__":
    main()
