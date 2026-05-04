#!/usr/bin/env python3
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
    parser.add_argument("--title", type=str, default=None, help="Optional chart title")
    parser.add_argument("--output", type=str, required=True, help="Output HTML path")
    return parser.parse_args()


def metric_key(metric: str) -> str:
    if metric == "delta":
        return "projection_delta_trait_minus_neutral"
    if metric == "trait":
        return "projection_score_trait"
    return "projection_score_neutral"


def collect_means(rows: list[dict[str, Any]], value_key: str) -> dict[str, float]:
    by_axis: dict[str, list[float]] = {}
    for row in rows:
        axis = row.get("projection_trait")
        if axis is None:
            continue
        by_axis.setdefault(str(axis), []).append(float(row[value_key]))
    return {axis: float(sum(vals) / len(vals)) for axis, vals in by_axis.items() if vals}


def axis_intersection(series: list[dict[str, float]]) -> list[str]:
    if not series:
        return []
    shared = set(series[0].keys())
    for s in series[1:]:
        shared &= set(s.keys())
    return sorted(shared)


def main() -> None:
    args = parse_args()
    if args.labels is not None and len(args.labels) != len(args.inputs):
        raise ValueError("--labels must match --inputs length")

    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise SystemExit("plotly is required. Install with: pip install plotly") from exc

    value_key = metric_key(args.metric)
    run_labels: list[str] = []
    run_means: list[dict[str, float]] = []

    for idx, input_path_str in enumerate(args.inputs):
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

    if args.top_k_axes is not None and args.axis_filter is None:
        strengths = []
        for axis in axes:
            avg_abs = float(np.mean([abs(run[axis]) for run in run_means]))
            strengths.append((axis, avg_abs))
        strengths.sort(key=lambda x: x[1], reverse=True)
        axes = [axis for axis, _ in strengths[: args.top_k_axes]]

    z = np.array([[run[axis] for axis in axes] for run in run_means], dtype=float)
    vmax = float(np.nanmax(np.abs(z))) if z.size else 1.0
    if vmax == 0.0:
        vmax = 1.0

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

