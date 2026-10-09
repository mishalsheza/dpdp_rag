"""Loading of YAML configuration files from configs/."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"


def load_config(name: str = "ingest.yaml", config_dir: Path = CONFIG_DIR) -> dict[str, Any]:
    """Read a YAML config file by name from the configs directory, or by path."""
    path = Path(name)
    if not path.is_file():
        path = config_dir / name
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Config {name} must be a mapping")
    return data


def merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge `overrides` into a copy of `base`."""
    out = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge(out[key], value)
        else:
            out[key] = value
    return out


def resolve(path: str | Path) -> Path:
    """Resolve a config path relative to the repository root."""
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p
