"""Altair line charts for the dashboard.

Colours are the first categorical slots of the dataviz reference palette, validated for
light and dark surfaces (adjacent CVD ΔE ≥ 24, normal-vision ΔE ≥ 31). Series keep a
fixed colour by name, lines are 2px with ~8px markers, every point has a tooltip, and
multi-series charts get a legend plus a direct label at the last point.
"""

from __future__ import annotations

from collections.abc import Sequence

import altair as alt
import pandas as pd

PALETTE = {
    "light": ["#2a78d6", "#eb6834", "#1baf7a"],
    "dark": ["#3987e5", "#d95926", "#199e70"],
}
TEXT = {"light": "#52514e", "dark": "#c3c2b7"}  # secondary text ink, never a series colour
SURFACE = {"light": "#fcfcfb", "dark": "#1a1a19"}  # ring around markers = chart surface


def _labels_collide(df: pd.DataFrame, x: str, y: str, series: str, min_gap: float) -> bool:
    """True if two series end within `min_gap` (share of the y range) of each other."""
    last = df.sort_values(x).groupby(series)[y].last().sort_values()
    span = float(df[y].max() - df[y].min()) or 1.0
    return bool((last.diff().dropna().abs() / span < min_gap).any())


def line_chart(
    df: pd.DataFrame,
    *,
    x: str,
    y: str,
    series: str | None,
    series_order: Sequence[str],
    y_title: str,
    y_format: str = "",
    mode: str = "light",
    x_title: str | None = None,
    tooltip_extra: Sequence[str] = (),
    height: int = 240,
    log_scale: bool = False,
) -> alt.Chart:
    mode = mode if mode in PALETTE else "light"
    colors = PALETTE[mode][: len(series_order)]
    multi = series is not None and len(series_order) > 1
    color = (
        alt.Color(
            f"{series}:N",
            scale=alt.Scale(domain=list(series_order), range=colors),
            legend=alt.Legend(orient="top", title=None, labelColor=TEXT[mode]) if multi else None,
        )
        if series
        else alt.value(colors[0])
    )
    x_enc = alt.X(
        f"{x}:T",
        title=x_title,
        axis=alt.Axis(grid=False, labelColor=TEXT[mode], titleColor=TEXT[mode]),
    )
    y_enc = alt.Y(
        f"{y}:Q",
        title=y_title,
        axis=alt.Axis(
            format=y_format,
            gridOpacity=0.25,
            labelColor=TEXT[mode],
            titleColor=TEXT[mode],
            # On a log scale keep only the powers of ten, so the grid stays recessive.
            values=[10**k for k in range(0, 7)] if log_scale else alt.Undefined,
        ),
        scale=alt.Scale(type="log") if log_scale else alt.Undefined,
    )
    tooltip = [alt.Tooltip(f"{x}:T", title="time")]
    if series:
        tooltip.append(alt.Tooltip(f"{series}:N", title="series"))
    tooltip += [alt.Tooltip(f"{y}:Q", title=y_title, format=y_format or ".3f")]
    tooltip += [alt.Tooltip(f"{c}") for c in tooltip_extra]

    base = alt.Chart(df).encode(x=x_enc, y=y_enc, color=color)
    hover = alt.selection_point(nearest=True, on="pointerover", fields=[x], empty=False)
    lines = base.mark_line(strokeWidth=2, interpolate="linear")
    # Large invisible hit targets carry the tooltip; visible points stay modest.
    targets = base.mark_point(size=400, opacity=0).encode(tooltip=tooltip).add_params(hover)
    points = base.mark_point(filled=True, size=64, stroke=SURFACE[mode], strokeWidth=2).encode(
        opacity=alt.condition(hover, alt.value(1), alt.value(0.85))
    )
    rule = (
        alt.Chart(df)
        .mark_rule(color=TEXT[mode], strokeWidth=1)
        .encode(x=f"{x}:T")
        .transform_filter(hover)
    )
    layers = [rule, lines, points, targets]
    if multi and not _labels_collide(df, x, y, series, 0.06):
        last = (
            base.transform_window(
                rank="rank()", sort=[alt.SortField(x, order="descending")], groupby=[series]
            )
            .transform_filter("datum.rank == 1")
            .mark_text(align="left", dx=8, fontSize=11)
            # Labels wear text ink; the encoding overrides the inherited series colour.
            .encode(text=f"{series}:N", color=alt.value(TEXT[mode]))
        )
        layers.append(last)
    return alt.layer(*layers).properties(height=height)
