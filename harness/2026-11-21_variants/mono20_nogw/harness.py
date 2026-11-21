"""Team Turquoise harness: score raw firm-year rows with the frozen model and parameters.

Command line (same as the 2026-11-16 test bundle):
    python harness.py --input_csv input.csv --output_csv predictions.csv
Python, one function call:
    from harness import predict
    pds = predict(df)          # df = pandas DataFrame or CSV path; returns an n x 1 numpy array
"""

import argparse
from pathlib import Path

from inference import predict, predict_csv  # noqa: F401  (predict is the one-call Python entry point)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_csv", "-input_csv", type=Path, required=True)
    parser.add_argument("--output_csv", "-output_csv", type=Path, required=True)
    args = parser.parse_args()
    if args.input_csv.resolve() == args.output_csv.resolve():
        parser.error("Input and output paths must differ.")
    predict_csv(args.input_csv, args.output_csv)


if __name__ == "__main__":
    main()
