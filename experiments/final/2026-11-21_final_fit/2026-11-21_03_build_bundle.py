"""Estimation step 3: copy the frozen model + parameters into the harness folder, write the checksum manifest,
and zip the runtime files. Fits nothing.

Usage: python 2026-11-21_03_build_bundle.py   (after 01 final and 02)
"""
import hashlib
import json
import shutil
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
BUNDLE = REPO / "harness/2026-11-21_turquoise_submission"
ZIP = REPO / "harness/2026-11-21_Turquoise_harness.zip"
RUNTIME = ["harness.py", "inference.py", "missing_rules.py", "anomaly_rules.py", "feature_rules.py",
           "bundle_paths.py", "README.md", "requirements.txt", "config/feature_schema.json",
           "data/interim/ateco_sector_english.csv", "artifacts/lightgbm.txt", "artifacts/preprocessing.json"]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    shutil.copyfile(HERE / "final_lightgbm.txt", BUNDLE / "artifacts/lightgbm.txt")
    params = json.loads((HERE / "final_params.json").read_text())
    schema = json.loads((BUNDLE / "config/feature_schema.json").read_text())
    canon = json.dumps(schema, sort_keys=True, separators=(",", ":"))
    assert hashlib.sha256(canon.encode()).hexdigest() == params["schema_hash"], "bundle schema != fitted schema"
    (BUNDLE / "artifacts/preprocessing.json").write_text(json.dumps(params, indent=2, allow_nan=False) + "\n")
    manifest = {
        "team": "Turquoise",
        "built": "2026-11-21",
        "model": "LightGBM, unconstrained, 210 trees, 75 features, fit on all labelled FY2007-2011 rows (835,016)",
        "level": "prior shift to tau = %.6f (D-072), floor %.4f, cap %.2f" % (
            params["calibration"]["tau"], params["calibration"]["floor"], params["calibration"]["cap"]),
        "source_code": "teammate/project-update @ 7949207 (schema 1.3.0) vendored in experiments/final/2026-11-21_final_fit",
        "bundle_sha256": {r: sha(BUNDLE / r) for r in RUNTIME},
        "feature_count": len(params["feature_columns"]),
        "schema_version": params["schema_version"],
        "invalid_feature_policy": "frozen fallback PD (tau) for any row whose features are not finite",
    }
    (BUNDLE / "artifacts/manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with ZipFile(ZIP, "w", compression=ZIP_DEFLATED) as z:
        for r in RUNTIME + ["artifacts/manifest.json"]:
            z.write(BUNDLE / r, r)  # files at the zip root, same layout as the 2026-11-16 zip
    print(f"bundle ready: {BUNDLE}\nzip: {ZIP} ({ZIP.stat().st_size/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
