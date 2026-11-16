"""Resolve resources within the standalone submission directory."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent


def project_path(relative: str) -> Path:
    return ROOT / relative
