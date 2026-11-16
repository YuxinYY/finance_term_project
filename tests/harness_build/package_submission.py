"""Create turquoise.zip and smoke-check its CLI outside the repository."""

import os
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "tests/harness_submission"
ARCHIVE = ROOT / "tests/turquoise.zip"


def main():
    with ZipFile(ARCHIVE, "w", compression=ZIP_DEFLATED) as archive:
        for path in sorted(SOURCE.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                archive.write(path, path.relative_to(SOURCE))
    with tempfile.TemporaryDirectory() as task_tmp:
        directory = Path(task_tmp)
        extracted = directory / "submission"
        with ZipFile(ARCHIVE) as archive:
            assert archive.testzip() is None
            archive.extractall(extracted)
        sample = pd.read_csv(ROOT / "data/raw/train_demo_subset.csv", nrows=128)
        sample = sample.drop(columns=["label", "def_date"], errors="ignore")
        input_csv = directory / "unlabeled input.csv"
        output_csv = directory / "predictions.csv"
        sample.to_csv(input_csv, index=False)
        environment = dict(os.environ)
        environment.pop("PYTHONPATH", None)
        environment["PYTHONNOUSERSITE"] = "1"
        for input_arg, output_arg in (("--input_csv", "--output_csv"),
                                      ("-input_csv", "-output_csv")):
            subprocess.run(
                [sys.executable, str(extracted / "harness.py"), input_arg,
                 str(input_csv), output_arg, str(output_csv)],
                cwd=directory, env=environment, check=True,
            )
            predictions = pd.read_csv(output_csv, header=None)
            assert predictions.shape == (len(sample), 1)
            assert np.isfinite(predictions.to_numpy()).all()
            assert predictions[0].between(0, 1).all()
        # Exercise the fallback using only files from the extracted archive.
        sample.loc[[0, 4, len(sample) - 1], "asst_tot"] = -1
        sample.to_csv(input_csv, index=False)
        subprocess.run(
            [sys.executable, str(extracted / "harness.py"), "--input_csv",
             str(input_csv), "--output_csv", str(output_csv)],
            cwd=directory, env=environment, check=True,
        )
        fallback_pd = json.loads((extracted / "artifacts/preprocessing.json").read_text())["fallback_pd"]
        predictions = pd.read_csv(output_csv, header=None)
        assert predictions.shape == (len(sample), 1)
        np.testing.assert_allclose(predictions.iloc[[0, 4, len(sample) - 1], 0], fallback_pd,
                                   rtol=0, atol=1e-16)
        assert np.isfinite(predictions.to_numpy()).all()
    print(f"PASS: standalone ZIP and both CLI argument styles; {ARCHIVE} ({ARCHIVE.stat().st_size:,} bytes).")


if __name__ == "__main__":
    main()
