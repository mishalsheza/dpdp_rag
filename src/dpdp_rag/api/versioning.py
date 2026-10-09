"""config_hash (sha256 of the config plus prompt files) and the deployed git SHA."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from dpdp_rag.config import REPO_ROOT
from dpdp_rag.generation.prompts import prompt_paths


def hash_config(config: dict[str, Any], files: dict[str, Path]) -> str:
    """sha256 over the canonical JSON of `config` and the bytes of each named file."""
    h = hashlib.sha256()
    h.update(json.dumps(config, sort_keys=True, default=str, separators=(",", ":")).encode())
    for name, path in sorted(files.items()):
        h.update(f"\0{name}\0".encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def config_hash(config: dict[str, Any]) -> str:
    """sha256 of the system config plus every prompt file it names.

    Any change to a setting or to a prompt changes the hash, so metrics, traces and
    cached responses can be attributed to the exact configuration that produced them.
    """
    return hash_config(config, prompt_paths(config))


def git_state() -> tuple[str, bool | None]:
    """(commit, dirty). $GIT_SHA wins (CI and containers; dirty unknown -> None). Dirty
    means `git status --porcelain` lists anything, untracked files included. Without git:
    ("unknown", None)."""
    if os.environ.get("GIT_SHA"):
        return os.environ["GIT_SHA"], None
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return "unknown", None
    return sha, bool(dirty.strip())


def git_sha() -> str:
    """$GIT_SHA if set, else `git rev-parse HEAD` (+ "-dirty" if changed), else "unknown"."""
    sha, dirty = git_state()
    return f"{sha}-dirty" if dirty else sha
