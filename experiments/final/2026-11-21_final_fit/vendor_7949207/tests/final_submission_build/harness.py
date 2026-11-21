"""Score an unlabeled raw CSV using the bundled frozen model and parameters."""

import argparse
from pathlib import Path

from inference import predict_csv


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
