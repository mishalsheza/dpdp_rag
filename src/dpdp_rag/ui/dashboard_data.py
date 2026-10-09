"""Data access for the dashboard: request metrics (SQLite) and eval run history (JSON)."""

from __future__ import annotations

import html
import json
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

from dpdp_rag.metrics.store import percentile

BUCKETS = {"hour": "h", "day": "D", "week": "W"}


def load_requests(db_path: Path) -> pd.DataFrame:
    """All rows of the `requests` table; empty frame if the database does not exist yet."""
    cols = [
        "timestamp",
        "config_hash",
        "latency_ms",
        "tokens_in",
        "tokens_out",
        "cost_usd",
        "cached",
        "refused",
        "source",
    ]
    if not db_path.exists():
        return pd.DataFrame(columns=cols)
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(f"SELECT {', '.join(cols)} FROM requests", conn)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, format="ISO8601")
    return df


def filter_requests(
    df: pd.DataFrame,
    source: str = "all",
    config_hashes: list[str] | None = None,
    include_cached: bool = True,
) -> pd.DataFrame:
    out = df
    if source != "all":
        out = out[out["source"] == source]
    if config_hashes:
        out = out[out["config_hash"].isin(config_hashes)]
    if not include_cached:
        out = out[out["cached"] == 0]
    return out


def summary(df: pd.DataFrame) -> dict[str, float | int | None]:
    lat = df["latency_ms"].astype(float).tolist()
    return {
        "requests": len(df),
        "p50_latency_ms": percentile(lat, 50),
        "p99_latency_ms": percentile(lat, 99),
        "cost_per_request_usd": float(df["cost_usd"].mean()) if len(df) else None,
        "refusal_rate": float(df["refused"].mean()) if len(df) else None,
    }


def over_time(df: pd.DataFrame, bucket: str = "day") -> pd.DataFrame:
    """Per time bucket: request count, p50/p99 latency, mean cost, refusal rate."""
    if df.empty:
        return pd.DataFrame(
            columns=[
                "bucket",
                "requests",
                "p50_latency_ms",
                "p99_latency_ms",
                "cost_per_request_usd",
                "refusal_rate",
            ]
        )
    keyed = df.assign(
        bucket=df["timestamp"].dt.tz_convert(None).dt.to_period(BUCKETS[bucket]).dt.start_time
    )
    rows = []
    for b, g in keyed.groupby("bucket", sort=True):
        s = summary(g)
        rows.append({"bucket": b, **{k: s[k] for k in s}})
    return pd.DataFrame(rows)


def load_eval_runs(runs_dir: Path) -> list[dict[str, Any]]:
    """Every results.json under runs_dir, oldest first (unreadable files are skipped)."""
    runs = []
    for path in sorted(runs_dir.glob("*/results.json")) if runs_dir.exists() else []:
        try:
            runs.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    return sorted(runs, key=lambda r: r.get("started_at", ""))


EVAL_SCORES = {
    "recall@5": ("retrieval", "recall@5"),
    "MRR": ("retrieval", "mrr"),
    "refusal correctness": ("answers", "refusal_correctness"),
    "faithfulness": ("answers", "faithfulness_mean"),
    "relevance": ("answers", "relevance_mean"),
}


def eval_history(runs: list[dict[str, Any]]) -> pd.DataFrame:
    """Long format: one row per (run, metric) with the overall score."""
    rows = []
    for run in runs:
        for metric, (group, key) in EVAL_SCORES.items():
            value = (run.get("overall", {}).get(group) or {}).get(key)
            if value is not None:
                rows.append(
                    {
                        "run_id": run["run_id"],
                        "started_at": pd.to_datetime(run["started_at"]).tz_convert(None),
                        "config_hash": run.get("config_hash", "")[:8],
                        "git": git_badge(run),
                        "judge": run.get("judge_model", "unknown"),
                        "metric": metric,
                        "value": float(value),
                        "n": run.get("n"),
                    }
                )
    return pd.DataFrame(
        rows,
        columns=["run_id", "started_at", "config_hash", "git", "judge", "metric", "value", "n"],
    )


# (column, results group, key) for the per-category table, in display order.
CATEGORY_COLUMNS = [
    ("recall@5", "retrieval", "recall@5"),
    ("MRR", "retrieval", "mrr"),
    ("faithfulness", "answers", "faithfulness_mean"),
    ("relevance", "answers", "relevance_mean"),
    ("refusal correctness", "answers", "refusal_correctness"),
    ("injection resisted", "answers", "injection_resisted"),
]
DASH = "—"


def _items(run: dict[str, Any], cat: str) -> list[dict[str, Any]]:
    return [i for i in run.get("items", []) if i.get("category") == cat]


def _judge_failed(items: list[dict[str, Any]], agg: dict[str, Any]) -> bool:
    if agg.get("answers", {}).get("judge_failed"):
        return True
    # Runs written before judge_failed existed: look for judge errors on the items.
    return any(e.startswith("judge ") for i in items for e in i.get("errors", []))


def missing_reason(run: dict[str, Any], cat: str, column: str) -> str:
    """Why a per-category metric is empty, in words for a tooltip."""
    agg = run.get("by_category", {}).get(cat, {})
    items = _items(run, cat)
    if not agg.get("n"):
        return "No questions in this category."
    group = dict((c, g) for c, g, _ in CATEGORY_COLUMNS)[column]
    if group == "retrieval":
        if items and all(not i.get("gold_chunk_ids") for i in items):
            return (
                "No gold chunks: these questions have no answer in the documents, so "
                "retrieval isn't scored."
            )
        return "Retrieval failed for every question in this category (see item errors)."
    if column == "injection resisted" and not any(
        i.get("category") == "prompt_injection" for i in items
    ):
        return "Injection resistance is scored only for prompt-injection questions."
    if column == "relevance" and items and not any(i.get("answerable") for i in items):
        return "Relevance is scored only for answerable questions."
    if items and all((i.get("answer") or {}).get("error") or not i.get("answer") for i in items):
        return "No answer was generated (answer-model errors), so nothing was judged."
    if _judge_failed(items, agg):
        return "The judge failed for this category; see the banner and item errors."
    if column == "faithfulness":
        return "Every answer was the fixed 'can't answer' refusal, which makes no claims."
    return "Not scored for this category."


def category_cells(run: dict[str, Any]) -> list[dict[str, Any]]:
    """One row per category: n and each metric as {"text", "tooltip"}. Values are
    rounded to 2 decimals; an empty metric is "—" with the reason as its tooltip."""
    rows = []
    for cat, agg in run.get("by_category", {}).items():
        row: dict[str, Any] = {"category": cat, "n": agg.get("n")}
        for column, group, key in CATEGORY_COLUMNS:
            value = (agg.get(group) or {}).get(key)
            row[column] = (
                {"text": f"{value:.2f}", "tooltip": None}
                if value is not None
                else {"text": DASH, "tooltip": missing_reason(run, cat, column)}
            )
        rows.append(row)
    return rows


def category_table_html(run: dict[str, Any]) -> str:
    """The per-category table as HTML; empty cells carry their reason in a title tooltip."""

    def cell(c: dict[str, Any]) -> str:
        if c["tooltip"] is None:
            return f'<td style="text-align:right">{c["text"]}</td>'
        tip = html.escape(c["tooltip"], quote=True)
        return (
            f'<td style="text-align:right"><span title="{tip}" '
            f'style="cursor:help;opacity:0.6">{DASH}</span></td>'
        )

    head = "".join(
        f'<th style="text-align:{"left" if i == 0 else "right"}">{h}</th>'
        for i, h in enumerate(["category", "n", *[c for c, _, _ in CATEGORY_COLUMNS]])
    )
    body = "".join(
        f"<tr><td>{html.escape(r['category'])}</td>"
        f'<td style="text-align:right">{r["n"]}</td>'
        + "".join(cell(r[c]) for c, _, _ in CATEGORY_COLUMNS)
        + "</tr>"
        for r in category_cells(run)
    )
    return f'<table style="width:100%"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


def category_table(run: dict[str, Any]) -> pd.DataFrame:
    """The per-category metrics as numbers (None where empty), rounded to 2 decimals."""
    rows = []
    for r in category_cells(run):
        rows.append(
            {
                "category": r["category"],
                "n": r["n"],
                **{
                    c: None if r[c]["tooltip"] is not None else float(r[c]["text"])
                    for c, _, _ in CATEGORY_COLUMNS
                },
            }
        )
    return pd.DataFrame(rows)


def git_badge(run: dict[str, Any]) -> str:
    """ "abc1234", "abc1234 · dirty" or "unknown" (runs before git was recorded, or CI)."""
    g = run.get("git") or {}
    sha = g.get("sha")
    if not sha or sha == "unknown":
        return "unknown"
    return f"{sha[:7]} · dirty" if g.get("dirty") else sha[:7]
