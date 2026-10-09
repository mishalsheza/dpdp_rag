"""Prompt files from prompts/, as named in `generation.prompt_files`."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Any

from dpdp_rag.config import resolve


def prompt_paths(config: dict[str, Any]) -> dict[str, Path]:
    """Every prompt file the system config uses (answering and query rewriting)."""
    paths = {name: resolve(p) for name, p in config["generation"]["prompt_files"].items()}
    rewrite = (config.get("query_rewrite") or {}).get("prompt_files") or {}
    paths.update({f"query_rewrite.{name}": resolve(p) for name, p in rewrite.items()})
    return paths


@dataclass(frozen=True)
class Prompts:
    system: str
    user: Template  # $as_of_date, $context, $question
    insufficient_context: str
    schema: dict[str, Any]

    @classmethod
    def load(cls, config: dict[str, Any]) -> Prompts:
        paths = prompt_paths(config)
        read = {name: p.read_text(encoding="utf-8") for name, p in paths.items()}
        return cls(
            system=read["system"].strip(),
            user=Template(read["user"]),
            insufficient_context=read["insufficient_context"].strip(),
            schema=json.loads(read["schema"]),
        )
