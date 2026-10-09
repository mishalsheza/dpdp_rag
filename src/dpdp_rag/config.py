"""Loading of YAML configuration files from configs/."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"


def load_config(name: str = "ingest.yaml", config_dir: Path = CONFIG_DIR) -> dict[str, Any]:
    """Read a YAML config file from the configs directory."""
    with (config_dir / name).open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Config {name} must be a mapping")
    return data


def resolve(path: str | Path) -> Path:
    """Resolve a config path relative to the repository root."""
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p
