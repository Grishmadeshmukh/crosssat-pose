from __future__ import annotations

import sys
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def add_src_to_path() -> None:
    src_root = project_root() / "src"
    src_root_str = str(src_root)
    if src_root_str not in sys.path:
        sys.path.insert(0, src_root_str)


def default_dataset_root() -> str:
    return str(project_root() / "spe3r")


def default_output_root(name: str) -> str:
    return str(project_root() / "outputs" / name)
