"""Watermark-noise removal: the India Code "IndiaCode" watermark and its fragments."""

from __future__ import annotations

import re
from typing import Any

import pymupdf
import pytest

from dpdp_rag.ingest.pdf import PageSettings, read_page
from dpdp_rag.ingest.text import is_watermark_noise, watermark_fragments

FRAGS = watermark_fragments(["IndiaCode"], 2)


@pytest.mark.parametrize("text", ["aC", "di", "In", "od", "e", "IndiaCode", "  aC  "])
def test_fragments_are_noise(text: str) -> None:
    assert is_watermark_noise(text, FRAGS)


@pytest.mark.parametrize("text", ["or", "and", "(a)", "In this Act", "1.", "", "Board."])
def test_real_text_is_kept(text: str) -> None:
    assert not is_watermark_noise(text, FRAGS)


def _settings(config: dict[str, Any]) -> PageSettings:
    return PageSettings.from_config(config["noise"], config["sources"]["act"])


def test_synthetic_page_drops_watermark_spans_and_fragment_lines(config: dict[str, Any]) -> None:
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 100), "4. Grounds for processing personal data.", fontsize=11)
    page.insert_text((72, 130), "aC", fontsize=11)  # stray fragment line
    page.insert_text((72, 160), "or", fontsize=11)  # a real one-word line
    page.insert_text((72, 190), "di", fontsize=11)
    page.insert_text((300, 400), "IndiaCode", fontsize=30, color=(150 / 255,) * 3)  # grey
    page.insert_text((200, 600), "IndiaCode", fontsize=30, rotate=90)  # rotated
    texts = [ln.text.strip() for ln in read_page(page, _settings(config)).lines]
    assert texts == ["4. Grounds for processing personal data.", "or"]


def test_act_pdf_has_no_watermark_residue(ingest, config: dict[str, Any]) -> None:
    act = [c for c in ingest.chunks if c.doc_type == "act"]
    for chunk in act:
        assert "IndiaCode" not in chunk.text
        # No isolated watermark fragment such as " aC " between words.
        assert not re.search(r"(?<!\S)(aC|di|nd|od|ia|Co|de)(?!\S)", chunk.text), chunk.chunk_id


def test_act_footnote_and_superscripts_removed(by_id) -> None:
    s1_2 = by_id["dpdp_act_2023:s1(2)"].text
    assert "on such date as the Central Government" in s1_2  # "date1" lost its marker
    assert "*" not in s1_2
    # The India Code footnote under section 2(j) is not part of the definition.
    assert "vide notifn" not in by_id["dpdp_act_2023:s2(j)"].text
    assert all("Eighteen Months from" not in c.text for c in by_id.values() if c.doc_type == "act")


def test_ordinal_superscripts_kept(by_id) -> None:
    assert "dated the 3rd January, 2025" in by_id["dpdp_rules_2025:preamble"].text
