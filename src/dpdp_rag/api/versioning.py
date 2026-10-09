"""config_hash (sha256 of the config plus prompt files) and the deployed git SHA."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from typing import Any

from dpdp_rag.config import REPO_ROOT
from dpdp_rag.generation.prompts import prompt_paths


def config_hash(config: dict[str, Any]) -> str:
    """sha256 over the canonical JSON of `config` and the bytes of every prompt file.

    Any change to a setting or to a prompt changes the hash, so metrics, traces and
    cached responses can be attributed to the exact configuration that produced them.
    """
    h = hashlib.sha256()
    h.update(json.dumps(config, sort_keys=True, default=str, separators=(",", ":")).encode())
    for name, path in sorted(prompt_paths(config).items()):
        h.update(f"\0{name}\0".encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def git_sha() -> str:
    """$GIT_SHA if set, else `git rev-parse HEAD` (+ "-dirty" if changed), else "unknown"."""
    if os.environ.get("GIT_SHA"):
        return os.environ["GIT_SHA"]
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
        return "unknown"
    return f"{sha}-dirty" if dirty.strip() else sha
