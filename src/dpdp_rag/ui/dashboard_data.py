"""Data access for the dashboard: request metrics (SQLite) and eval run history (JSON)."""

from __future__ import annotations

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
                        "metric": metric,
                        "value": float(value),
                        "n": run.get("n"),
                    }
                )
    return pd.DataFrame(
        rows, columns=["run_id", "started_at", "config_hash", "metric", "value", "n"]
    )


def category_table(run: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for cat, agg in run.get("by_category", {}).items():
        r, a = agg.get("retrieval", {}), agg.get("answers", {})
        rows.append(
            {
                "category": cat,
                "n": agg.get("n"),
                "recall@5": r.get("recall@5"),
                "MRR": r.get("mrr"),
                "faithfulness": a.get("faithfulness_mean"),
                "relevance": a.get("relevance_mean"),
                "refusal correctness": a.get("refusal_correctness"),
            }
        )
    return pd.DataFrame(rows)
