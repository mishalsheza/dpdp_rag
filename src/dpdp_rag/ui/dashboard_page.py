"""Dashboard page: latency, cost and refusal trends from metrics.db, plus eval history."""

from __future__ import annotations

import streamlit as st

from dpdp_rag.config import load_system_config, resolve
from dpdp_rag.eval.report import judge_banner
from dpdp_rag.ui import dashboard_data as dd
from dpdp_rag.ui.charts import line_chart

config = load_system_config()
mode = "dark" if (st.context.theme.type or "light") == "dark" else "light"
db_path = resolve(config["metrics"]["db_path"])
runs_dir = resolve(config["ui"]["eval_runs_dir"])

st.title("Metrics")
requests = dd.load_requests(db_path)

# Filters, in one row above the charts.
c1, c2, c3, c4 = st.columns([1, 2, 1, 1])
source = c1.selectbox("Source", ["all", "api", "eval"])
hashes = sorted(requests["config_hash"].unique().tolist()) if len(requests) else []
chosen = c2.multiselect(
    "Config hash", hashes, format_func=lambda h: h[:12], placeholder="All configurations"
)
bucket = c3.selectbox("Bucket", list(dd.BUCKETS), index=1)
include_cached = c4.toggle("Include cached", value=True)

df = dd.filter_requests(requests, source, chosen, include_cached)
s = dd.summary(df)


def fmt(v: float | None, spec: str) -> str:
    return "–" if v is None else format(v, spec)


t1, t2, t3, t4, t5 = st.columns(5)
t1.metric("Requests", f"{s['requests']:,}")
t2.metric("p50 latency", fmt(s["p50_latency_ms"], ",.0f") + " ms")
t3.metric("p99 latency", fmt(s["p99_latency_ms"], ",.0f") + " ms")
t4.metric("Cost / request", "$" + fmt(s["cost_per_request_usd"], ".5f"))
t5.metric("Refusal rate", fmt(s["refusal_rate"], ".0%"))

if df.empty:
    st.info(
        f"No requests recorded in `{db_path}` for these filters yet. Use the chat page, "
        "run the load test, or run `uv run dpdp-eval run`."
    )
else:
    ts = dd.over_time(df, bucket)
    lat = ts.melt(
        id_vars=["bucket", "requests"],
        value_vars=["p50_latency_ms", "p99_latency_ms"],
        var_name="series",
        value_name="ms",
    )
    lat["series"] = lat["series"].map({"p50_latency_ms": "p50", "p99_latency_ms": "p99"})
    st.subheader("Latency")
    log_scale = st.toggle(
        "Log scale", value=False, help="Useful when a cold start dwarfs normal latency."
    )
    st.altair_chart(
        line_chart(
            lat,
            x="bucket",
            y="ms",
            series="series",
            series_order=["p50", "p99"],
            y_title="Latency (ms)",
            y_format=",.0f",
            mode=mode,
            tooltip_extra=["requests"],
            log_scale=log_scale,
        ),
        width="stretch",
    )
    left, right = st.columns(2)
    with left:
        st.subheader("Cost per request")
        st.altair_chart(
            line_chart(
                ts,
                x="bucket",
                y="cost_per_request_usd",
                series=None,
                series_order=["cost"],
                y_title="USD",
                y_format="$.5f",
                mode=mode,
                tooltip_extra=["requests"],
            ),
            width="stretch",
        )
    with right:
        st.subheader("Refusal rate")
        st.altair_chart(
            line_chart(
                ts,
                x="bucket",
                y="refusal_rate",
                series=None,
                series_order=["refusal rate"],
                y_title="Share refused",
                y_format=".0%",
                mode=mode,
                tooltip_extra=["requests"],
            ),
            width="stretch",
        )
    with st.expander("Table view"):
        st.dataframe(ts, hide_index=True, width="stretch")

st.divider()
st.title("Eval history")
runs = dd.load_eval_runs(runs_dir)
if not runs:
    st.info(f"No eval runs found in `{runs_dir}`. Run `uv run dpdp-eval run`.")
else:
    hist = dd.eval_history(runs)
    unit = ["recall@5", "MRR", "refusal correctness"]
    scale5 = ["faithfulness", "relevance"]
    left, right = st.columns(2)
    with left:
        st.subheader("Retrieval & refusal (0–1)")
        st.altair_chart(
            line_chart(
                hist[hist["metric"].isin(unit)],
                x="started_at",
                y="value",
                series="metric",
                series_order=unit,
                y_title="Score",
                y_format=".2f",
                mode=mode,
                tooltip_extra=["run_id", "n"],
            ),
            width="stretch",
        )
    with right:
        st.subheader("Answer quality (judge, 1–5)")
        st.altair_chart(
            line_chart(
                hist[hist["metric"].isin(scale5)],
                x="started_at",
                y="value",
                series="metric",
                series_order=scale5,
                y_title="Mean score",
                y_format=".2f",
                mode=mode,
                tooltip_extra=["run_id", "n"],
            ),
            width="stretch",
        )
    latest = runs[-1]
    st.subheader(f"Latest run by category: `{latest['run_id']}` (n={latest.get('n')})")
    if banner := judge_banner(latest):
        st.error(banner, icon="⚠️")
    st.dataframe(dd.category_table(latest), hide_index=True, width="stretch")
    with st.expander("Table view of all runs"):
        st.dataframe(
            hist.pivot_table(
                index=["started_at", "run_id"], columns="metric", values="value"
            ).reset_index(),
            hide_index=True,
            width="stretch",
        )
