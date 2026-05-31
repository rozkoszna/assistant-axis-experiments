#!/usr/bin/env python3
"""
Interactive heatmap (HTML): user traits × projection axes.

Same layout as plot_traits_axes_heatmap.py but rendered with Plotly as a
self-contained HTML file — hover over any cell to see the exact value,
zoom in, and pan. Use this when you need to explore a large trait × axis
matrix where the static PNG becomes too dense to read.

Rows    = user traits (one input JSONL file per trait)
Columns = personality axes
Colour  = chosen metric, symmetric around zero (red = positive, blue = negative)

Metric choices:
  delta   — trait_score - neutral_score  (how much the trait shifts the model)
  trait   — raw projection score under the trait condition
  neutral — raw projection score under the neutral condition

Requires: plotly  (pip install plotly)
Input:    one projection JSONL per trait run
Output:   one standalone HTML file
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from plot_utils import infer_run_label, load_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an interactive Plotly heatmap for user traits (rows) vs projection axes (columns)."
    )
    parser.add_argument("--inputs", type=str, nargs="+", required=True, help="Projection JSONL files")
    parser.add_argument("--labels", type=str, nargs="+", default=None, help="Optional labels for inputs")
    parser.add_argument(
        "--metric",
        type=str,
        choices=["delta", "trait", "neutral"],
        default="delta",
        help="Heatmap value metric",
    )
    parser.add_argument("--top-k-axes", type=int, default=None, help="Keep only top K strongest axes")
    parser.add_argument("--axis-filter", type=str, nargs="+", default=None, help="Optional explicit axis order")
    parser.add_argument(
        "--mixed-only",
        action="store_true",
        help="Filter to axes where positive-fraction across traits is between 20%% and 80%%",
    )
    parser.add_argument(
        "--mixed-threshold",
        type=float,
        default=0.2,
        help="Fraction boundary for --mixed-only (default 0.2 means keep axes with 20%%-80%% positive)",
    )
    parser.add_argument("--title", type=str, default=None, help="Optional chart title")
    parser.add_argument("--output", type=str, required=True, help="Output HTML path")
    parser.add_argument(
        "--clip-pct",
        type=float,
        default=95.0,
        help="Clip color scale at this percentile of |z| (default 95 — prevents outliers bleaching the palette)",
    )
    return parser.parse_args()


def metric_key(metric: str) -> str:
    """Map the short metric name to the JSONL field name."""
    if metric == "delta":
        return "projection_delta_trait_minus_neutral"
    if metric == "trait":
        return "projection_score_trait"
    return "projection_score_neutral"


def collect_means(rows: list[dict[str, Any]], value_key: str) -> dict[str, float]:
    """Return mean value per axis for one trait run."""
    by_axis: dict[str, list[float]] = {}
    for row in rows:
        axis = row.get("projection_trait")
        if axis is None:
            continue
        by_axis.setdefault(str(axis), []).append(float(row[value_key]))
    return {axis: float(sum(vals) / len(vals)) for axis, vals in by_axis.items() if vals}


def axis_intersection(series: list[dict[str, float]]) -> list[str]:
    """Return only axes present in every trait run, so the matrix has no gaps."""
    if not series:
        return []
    shared = set(series[0].keys())
    for s in series[1:]:
        shared &= set(s.keys())
    return sorted(shared)


def main() -> None:
    args = parse_args()
    if args.labels is not None and len(args.labels) != len(
        [p for p in args.inputs if not Path(p).stem.endswith("__neutral")]
    ):
        raise ValueError("--labels must match --inputs length (excluding __neutral files)")

    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise SystemExit("plotly is required. Install with: pip install plotly") from exc

    value_key = metric_key(args.metric)
    run_labels: list[str] = []
    run_means: list[dict[str, float]] = []

    # Skip __neutral files — they are duplicates of the trait files (same delta rows).
    inputs = [p for p in args.inputs if not Path(p).stem.endswith("__neutral")]

    for idx, input_path_str in enumerate(inputs):
        path = Path(input_path_str)
        label = args.labels[idx] if args.labels is not None else infer_run_label(path)
        rows = load_jsonl(path)
        if not rows:
            raise ValueError(f"No rows in {path}")
        run_labels.append(label)
        run_means.append(collect_means(rows, value_key))

    if args.axis_filter:
        axes = list(dict.fromkeys(args.axis_filter))
    else:
        axes = axis_intersection(run_means)
    if not axes:
        raise ValueError("No shared axes found across inputs")

    if args.mixed_only and args.axis_filter is None:
        lo, hi = args.mixed_threshold, 1.0 - args.mixed_threshold
        mixed = []
        for axis in axes:
            vals = [run[axis] for run in run_means]
            pos_frac = sum(1 for v in vals if v > 0) / len(vals)
            if lo <= pos_frac <= hi:
                mixed.append(axis)
        axes = mixed
        if not axes:
            raise ValueError(f"No mixed axes found with threshold {args.mixed_threshold}")

    if args.top_k_axes is not None and args.axis_filter is None:
        # Rank axes by average absolute value across all traits, keep the strongest K.
        strengths = []
        for axis in axes:
            avg_abs = float(np.mean([abs(run[axis]) for run in run_means]))
            strengths.append((axis, avg_abs))
        strengths.sort(key=lambda x: x[1], reverse=True)
        axes = [axis for axis, _ in strengths[: args.top_k_axes]]

    z = np.array([[run[axis] for axis in axes] for run in run_means], dtype=float)
    # Clip colour scale at percentile to prevent outlier cells from bleaching the palette.
    vmax = float(np.nanpercentile(np.abs(z), args.clip_pct)) if z.size else 1.0
    if vmax == 0.0:
        vmax = float(np.nanmax(np.abs(z))) or 1.0

    metric_title = {"delta": "trait - neutral", "trait": "trait score", "neutral": "neutral score"}[args.metric]
    title = args.title or f"User traits × axes heatmap ({metric_title})"

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=axes,
            y=run_labels,
            colorscale="RdBu",
            zmid=0.0,
            zmin=-vmax,
            zmax=vmax,
            colorbar={"title": metric_title},
            hovertemplate="Trait: %{y}<br>Axis: %{x}<br>Value: %{z:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Projection axis",
        yaxis_title="User trait",
        width=max(1400, int(len(axes) * 28)),
        height=max(900, int(len(run_labels) * 24)),
    )
    fig.update_xaxes(tickangle=90)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Saved interactive heatmap to {out}")


if __name__ == "__main__":
    main()

