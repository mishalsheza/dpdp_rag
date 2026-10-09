"""The `python -m dpdp_rag.retrieval "query"` command line."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from dpdp_rag.retrieval.__main__ import main


def _write(cfg: dict[str, Any], tmp_path: Path) -> str:
    path = tmp_path / "retrieval.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return str(path)


def test_cli_json_with_filters_and_cross_refs(retrieval_config, tmp_path, capsys) -> None:
    config = _write(retrieval_config, tmp_path)
    main(
        [
            "exemption section 9 Fourth Schedule",
            "-k",
            "1",
            "--config",
            config,
            "--rule",
            "12",
            "--cross-refs",
            "--json",
        ]
    )
    out = json.loads(capsys.readouterr().out)
    assert [r["chunk_id"] for r in out] == [
        "dpdp_rules_2025:r12(1)",
        "dpdp_act_2023:s9(1)",
        "dpdp_act_2023:s9(3)",
    ]
    assert out[1]["expanded_from"] == "dpdp_rules_2025:r12(1)"
    assert out[0]["chunk"]["rule"] == "12"


def test_cli_text_output_and_in_force(retrieval_config, tmp_path, capsys) -> None:
    config = _write(retrieval_config, tmp_path)
    main(
        [
            "Consent Manager registration",
            "--config",
            config,
            "--mode",
            "bm25",
            "--in-force",
            "2026-12-01",
        ]
    )
    out = capsys.readouterr().out
    assert "dpdp_rules_2025:r4(1)" in out
    assert "in force 2026-11-13" in out
    assert "2027-05-13" not in out


def test_cli_runs_as_module(retrieval_config, tmp_path) -> None:
    config = _write(retrieval_config, tmp_path)
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "dpdp_rag.retrieval",
            "Department of the Central Government",
            "--config",
            config,
            "--mode",
            "bm25",
            "-k",
            "1",
            "--json",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(proc.stdout)[0]["chunk_id"] in {"dpdp_rules_2025:r13(5)", "gsr_892e:(ii)"}
