"""Developer utility: synchronize standalone rules from the current project.

Run this before building a new submission. Inference never executes notebooks.
"""

from __future__ import annotations

import ast
import hashlib
import json
import shutil
from pathlib import Path


DESTINATION = Path(__file__).resolve().parent
PROJECT = DESTINATION.parents[1]


def synchronize() -> None:
    sources = {}
    for notebook, filename in (("01_flag_anomalies", "anomaly_rules.py"),
                               ("02_missing_values", "missing_rules.py")):
        path = PROJECT / f"notebooks/features/{notebook}.ipynb"
        content = json.loads(path.read_text(encoding="utf-8"))
        exports = ["".join(c["source"]) for c in content["cells"]
                   if "export" in c.get("metadata", {}).get("tags", [])]
        if not exports:
            raise ValueError(f"No exported rules in {path}")
        source = "\n\n".join(exports)
        compile(source, str(path), "exec")
        (DESTINATION / filename).write_text(
            f'"""Generated from {path.relative_to(PROJECT)}; refresh with sync_rules.py."""\n'
            "import numpy as np\nimport pandas as pd\nfrom bundle_paths import project_path\n\n"
            + source + "\n", encoding="utf-8",
        )
        sources[str(path.relative_to(PROJECT))] = hashlib.sha256(path.read_bytes()).hexdigest()

    # The feature notebooks contain exploratory/legacy definitions. The schema
    # and the actual training pipeline define the current model feature contract.
    path = PROJECT / "src/finance_project/data_processing/preprocess.py"
    source = path.read_text(encoding="utf-8")
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
        raise ValueError("Required training feature definitions were renamed or removed.")
    (DESTINATION / "feature_rules.py").write_text(
        '"""Generated from the current schema and preprocess.py; refresh with sync_rules.py."""\n'
        "from __future__ import annotations\nimport json\nimport numpy as np\nimport pandas as pd\n"
        "from bundle_paths import project_path\n"
        'FEATURE_SCHEMA = json.loads(project_path("config/feature_schema.json").read_text())\n\n'
        + "\n\n".join(extracted) + "\n", encoding="utf-8",
    )
    sources[str(path.relative_to(PROJECT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    for relative in ("config/feature_schema.json", "data/interim/ateco_sector_english.csv"):
        path = PROJECT / relative
        destination = DESTINATION / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        sources[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    (DESTINATION / "rules_provenance.json").write_text(
        json.dumps(sources, indent=2) + "\n", encoding="utf-8",
    )
    print(f"Synchronized notebook rules and schema into {DESTINATION}.")


if __name__ == "__main__":
    synchronize()
