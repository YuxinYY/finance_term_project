"""Resolve bundled resources independently of the current working directory."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent


def project_path(relative: str) -> Path:
    return ROOT / relative
