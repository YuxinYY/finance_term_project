"""Replace generated sections in the Markdown report with current JSON artifacts."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from finance_project.common import project_path, read_json


def replace_generated_block(document: str, name: str, content: str) -> str:
    start = f"<!-- GENERATED:{name}:START -->"
    end = f"<!-- GENERATED:{name}:END -->"
    pattern = re.compile(f"({re.escape(start)}).*?({re.escape(end)})", re.DOTALL)
    if not pattern.search(document):
        raise ValueError(f"Could not find generated block '{name}' in report template.")
    return pattern.sub(lambda match: f"{match.group(1)}\n{content}\n{match.group(2)}", document)


def data_summary_markdown(summary: dict) -> str:
    lines = [
        "| Item | Value |",
        "| --- | ---: |",
        f"| Rows after cleaning | {summary['rows']:,} |",
        f"| Feature columns | {summary['columns']:,} |",
        f"| Duplicate rows removed | {summary['duplicate_rows_removed']:,} |",
        f"| Missing cells | {summary['missing_cells']:,} |",
    ]
    dropped = ", ".join(summary["dropped_index_columns"]) or "None"
    lines.append(f"| Dropped CSV index columns | {dropped} |")
    if summary["top_missing_columns"]:
        lines.extend(["", "Most incomplete fields:", "", "| Column | Missing | Missing % |", "| --- | ---: | ---: |"])
        lines.extend(
            f"| {item['column']} | {item['missing']:,} | {item['missing_pct']:.2f}% |"
            for item in summary["top_missing_columns"]
        )
    return "\n".join(lines)


def evaluation_markdown(result: dict) -> str:
    lines = [
        f"Held-out evaluation for `{result['model']}` predicting `{result['target']}` "
        f"({result['task']}; n = {result['evaluation_rows']:,}).",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    lines.extend(f"| {name} | {value:.6f} |" for name, value in result["metrics"].items())
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", default="reports/report.md")
    parser.add_argument("--data-summary", default="reports/metrics/data_summary.json")
    parser.add_argument("--evaluation", default="reports/metrics/model_evaluation.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_path: Path = project_path(args.report)
    report = report_path.read_text(encoding="utf-8")
    data_summary_path = project_path(args.data_summary)
    evaluation_path = project_path(args.evaluation)
    if data_summary_path.exists():
        report = replace_generated_block(report, "DATA_SUMMARY", data_summary_markdown(read_json(data_summary_path)))
    else:
        print(f"Skipping data section; artifact not found: {data_summary_path}")
    if evaluation_path.exists():
        report = replace_generated_block(report, "MODEL_METRICS", evaluation_markdown(read_json(evaluation_path)))
    else:
        print(f"Skipping model-metrics section; artifact not found: {evaluation_path}")
    report_path.write_text(report, encoding="utf-8")
    print(f"Updated report: {report_path}")


if __name__ == "__main__":
    main()
