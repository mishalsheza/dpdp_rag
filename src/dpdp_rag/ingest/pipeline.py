"""End-to-end ingestion: PDFs in data/raw -> data/processed/chunks.jsonl."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from dpdp_rag.config import load_config, resolve
from dpdp_rag.ingest.act import parse_act
from dpdp_rag.ingest.corrigendum import Patch, apply_patches, parse_corrigendum, patch_chunks
from dpdp_rag.ingest.in_force import (
    ActCommencement,
    parse_act_commencement,
    parse_gazette_date,
    parse_rules_commencement,
)
from dpdp_rag.ingest.models import Chunk
from dpdp_rag.ingest.notifications import parse_notification
from dpdp_rag.ingest.pdf import PageSettings
from dpdp_rag.ingest.refs import extract_refs, refers_to
from dpdp_rag.ingest.rules import parse_rules

log = logging.getLogger(__name__)


@dataclass
class IngestResult:
    chunks: list[Chunk]
    patches: list[Patch]
    act_commencement: ActCommencement
    rules_commencement: dict[str, date]


def _check_not_excluded(source: dict[str, Any], excluded: set[str]) -> None:
    if source["file"] in excluded:
        raise ValueError(f"{source['file']} is configured as excluded but listed as a source")


def _set_refs(chunk: Chunk, max_section: int, extra: str = "") -> None:
    refs = extract_refs(f"{chunk.text} {extra}".strip(), max_section=max_section)
    chunk.refers_to, chunk.refers_to_provisions = refers_to(refs)


def _attach_penalties(act_chunks: list[Chunk]) -> None:
    """Copy each Schedule penalty onto the section chunks the Schedule row cites."""
    rows = [c for c in act_chunks if c.schedule and c.penalty]
    sections = [c for c in act_chunks if c.section]
    for row in rows:
        breach = row.text.split(" Penalty: ")[0]  # refs in the penalty column are not breaches
        for ref in extract_refs(breach):
            for c in sections:
                if c.section == ref.section and ref.sub_section in (None, c.sub_section):
                    c.penalty = row.penalty


def build(config: dict[str, Any]) -> IngestResult:
    noise, sources = config["noise"], config["sources"]
    raw = resolve(config["paths"]["raw_dir"])
    excluded = {e["file"] for e in config.get("excluded", [])}
    act_src, rules_src, corr_src = sources["act"], sources["rules"], sources["corrigendum"]
    max_section = int(act_src["max_section"])

    def settings(src: dict[str, Any]) -> PageSettings:
        _check_not_excluded(src, excluded)
        return PageSettings.from_config(noise, src)

    # Notifications; G.S.R. 843(E) yields the Act's commencement dates.
    notif_chunks: list[Chunk] = []
    commencement: ActCommencement | None = None
    for src in sources["notifications"]:
        parsed = parse_notification(raw / src["file"], src, settings(src))
        notif_chunks.extend(parsed.chunks)
        if src.get("act_commencement"):
            clauses = [c.text for c in parsed.chunks if c.clause and c.clause.isalpha()]
            commencement = parse_act_commencement(
                clauses,
                parsed.published,
                fallback=act_src["enactment_date"],
                max_section=max_section,
            )
    if commencement is None:
        raise ValueError("No notification is flagged act_commencement in the config")

    # The Act.
    act_chunks = parse_act(
        raw / act_src["file"],
        act_src,
        settings(act_src),
        commencement,
        int(config["max_chunk_chars"]),
    )
    for c in act_chunks:
        if c.schedule:
            c.in_force_date = commencement.date_for("33")  # the Schedule is applied by section 33
        elif c.section is None:
            c.in_force_date = act_src["enactment_date"]
        else:
            c.in_force_date = commencement.date_for(c.section, c.sub_section, c.clause)
            if c.in_force_date == commencement.fallback:
                log.warning("%s is not named in G.S.R. 843(E); using enactment date", c.chunk_id)
        _set_refs(c, max_section)
    _attach_penalties(act_chunks)

    # The Rules, then the corrigendum applied to them.
    rules = parse_rules(raw / rules_src["file"], rules_src, settings(rules_src))
    corr = parse_corrigendum(raw / corr_src["file"], corr_src, settings(corr_src))
    target_chunks = [c for c in rules.chunks if c.gsr_no == corr_src["target_gsr_no"]]
    apply_patches(corr.patches, target_chunks, corr_src["gsr_no"])

    published = parse_gazette_date(rules.published_line)
    rule1 = [c.text for c in rules.chunks if c.rule == "1" and c.sub_rule]
    rule_dates = parse_rules_commencement(rule1, published)
    for c in rules.chunks:
        if c.rule:
            c.in_force_date = rule_dates[c.rule]
        elif c.schedule:
            c.see_rules = list(rules.schedule_rules[c.chunk_id])
            dates = [rule_dates[r] for r in c.see_rules]
            if len(set(dates)) > 1:
                log.warning(
                    "%s: schedule cited by rules with different dates; using earliest", c.chunk_id
                )
            c.in_force_date = min(dates)
        else:
            c.in_force_date = published
        _set_refs(c, max_section, rules.schedule_context.get(c.chunk_id, ""))

    for c in notif_chunks:
        _set_refs(c, max_section)
    by_id = {c.chunk_id: c for c in rules.chunks}
    corr_chunks = patch_chunks(corr, corr_src, by_id)

    chunks = act_chunks + rules.chunks + notif_chunks + corr_chunks
    ids = [c.chunk_id for c in chunks]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate chunk_id values")
    return IngestResult(chunks, corr.patches, commencement, rule_dates)


def write_outputs(result: IngestResult, config: dict[str, Any]) -> tuple[Path, Path]:
    paths = config["paths"]
    chunks_path, patches_path = resolve(paths["chunks_file"]), resolve(paths["patches_file"])
    chunks_path.parent.mkdir(parents=True, exist_ok=True)
    with chunks_path.open("w", encoding="utf-8") as fh:
        for chunk in result.chunks:
            fh.write(chunk.model_dump_json() + "\n")
    patches = [
        p.model_dump(
            include={
                "page",
                "line",
                "line_end",
                "old_text",
                "new_text",
                "target_rule_or_schedule",
                "item",
                "kind",
                "target_chunk_ids",
                "match_strategy",
                "source_text",
            }
        )
        for p in result.patches
    ]
    patches_path.write_text(
        json.dumps(patches, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return chunks_path, patches_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="ingest.yaml", help="config file name in configs/")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config = load_config(args.config)
    result = build(config)
    chunks_path, patches_path = write_outputs(result, config)
    log.info(
        "Wrote %d chunks to %s and %d patches to %s",
        len(result.chunks),
        chunks_path,
        len(result.patches),
        patches_path,
    )


if __name__ == "__main__":
    main()
