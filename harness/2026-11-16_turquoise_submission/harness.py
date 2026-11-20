"""Predict PDs using the frozen temporal_firm_year_v2 round-03 pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from inference import predict_csv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_csv", "-input_csv", required=True, type=Path)
    parser.add_argument("--output_csv", "-output_csv", required=True, type=Path)
    args = parser.parse_args()
    if args.input_csv.resolve() == args.output_csv.resolve():
        parser.error("Input and output paths must differ.")
    predict_csv(args.input_csv, args.output_csv)


if __name__ == "__main__":
    main()
