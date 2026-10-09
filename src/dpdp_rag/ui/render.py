"""Pure helpers for the chat page: API calls and citation badges (no Streamlit here)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from typing import Any

import httpx

REFUSAL_LABELS = {
    "insufficient_context": "The documents don't support an answer to this question.",
    "legal_advice": "This is not legal advice: the answer explains what the text says but "
    "does not assess your situation.",
    "model_refusal": "The model declined to answer.",
}


@dataclass(frozen=True)
class Badge:
    label: str
    color: str  # a Streamlit badge colour name
    icon: str


def api_url(ui_cfg: dict[str, Any]) -> str:
    return os.environ.get("DPDP_API_URL") or ui_cfg["api_url"]


def ask(base_url: str, question: str, as_of: date, timeout: float) -> dict[str, Any]:
    """POST /ask; raises RuntimeError with the API's message on failure."""
    try:
        r = httpx.post(
            f"{base_url.rstrip('/')}/ask",
            json={"question": question, "as_of_date": as_of.isoformat()},
            timeout=timeout,
        )
    except httpx.HTTPError as exc:
        raise RuntimeError(f"Could not reach the API at {base_url}: {exc}") from exc
    if r.status_code != 200:
        try:
            detail = r.json().get("detail", r.text)
        except ValueError:
            detail = r.text
        raise RuntimeError(f"API error {r.status_code}: {detail}")
    return r.json()


def version(base_url: str) -> dict[str, Any] | None:
    try:
        r = httpx.get(f"{base_url.rstrip('/')}/version", timeout=5)
        return r.json() if r.status_code == 200 else None
    except httpx.HTTPError:
        return None


def citation_badges(citation: dict[str, Any], as_of: date) -> list[Badge]:
    """Badges for "in force from <date>" (or not yet in force) and "corrected by G.S.R. 892(E)"."""
    badges: list[Badge] = []
    start = citation.get("in_force_date")
    if start:
        if citation.get("in_force"):
            badges.append(Badge(f"In force from {start}", "green", ":material/check_circle:"))
        else:
            badges.append(
                Badge(
                    f"In force from {start} · not yet on {as_of.isoformat()}",
                    "orange",
                    ":material/schedule:",
                )
            )
    if citation.get("corrected_by"):
        badges.append(
            Badge(f"Corrected by {citation['corrected_by']}", "violet", ":material/edit_note:")
        )
    return badges


def badge_markdown(badges: list[Badge]) -> str:
    """Streamlit markdown badges, e.g. ':green-badge[:material/check_circle: In force ...]'."""
    return " ".join(f":{b.color}-badge[{b.icon} {b.label}]" for b in badges)


def source_line(citation: dict[str, Any]) -> str:
    parts = [citation.get("gsr_no") or "DPDP Act, 2023"]
    if citation.get("page"):
        parts.append(f"p. {citation['page']}")
    parts.append(f"`{citation['chunk_id']}`")
    return " · ".join(parts)


def footer(response: dict[str, Any]) -> str:
    tokens = response.get("tokens", {}).get("total", 0)
    bits = [
        f"{response['latency_ms']:.0f} ms",
        f"${response['cost_usd']:.5f}",
        f"{tokens} tokens",
        f"model `{response['model']}`",
        f"config `{response['config_hash'][:8]}`",
    ]
    if response.get("cached"):
        bits.append("cached")
    return " · ".join(bits)
