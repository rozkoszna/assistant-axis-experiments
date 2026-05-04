#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from plot_utils import infer_run_label, load_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot a heatmap of user traits (rows) vs projection axes (columns)."
    )
    parser.add_argument("--inputs", type=str, nargs="+", required=True, help="Projection JSONL files")
    parser.add_argument(
        "--labels",
        type=str,
        nargs="+",
        default=None,
        help="Optional labels for inputs. Defaults to inferred trait labels.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        choices=["delta", "trait", "neutral"],
        default="delta",
        help="Heatmap value: trait-neutral delta, trait score, or neutral score.",
    )
    parser.add_argument(
        "--top-k-axes",
        type=int,
        default=None,
        help="Keep only top K axes by average absolute value across traits.",
    )
    parser.add_argument(
        "--axis-filter",
        type=str,
        nargs="+",
        default=None,
        help="Optional explicit axis list and order.",
    )
    parser.add_argument("--title", type=str, default=None, help="Optional plot title")
    parser.add_argument("--dpi", type=int, default=300, help="Output DPI")
    parser.add_argument("--output", type=str, required=True, help="Output PNG path")
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

    value_key = metric_key(args.metric)
    run_labels: list[str] = []
    run_means: list[dict[str, float]] = []

    for idx, input_path_str in enumerate(args.inputs):
        input_path = Path(input_path_str)
        label = args.labels[idx] if args.labels is not None else infer_run_label(input_path)
        rows = load_jsonl(input_path)
        if not rows:
            raise ValueError(f"No rows in {input_path}")
        run_labels.append(label)
        run_means.append(collect_means(rows, value_key))

    if args.axis_filter:
        axes = list(dict.fromkeys(args.axis_filter))
    else:
        axes = axis_intersection(run_means)

    if not axes:
        raise ValueError("No shared axes found across inputs")

    if args.top_k_axes is not None and args.axis_filter is None:
        strengths: list[tuple[str, float]] = []
        for axis in axes:
            avg_abs = float(np.mean([abs(run[axis]) for run in run_means]))
            strengths.append((axis, avg_abs))
        strengths.sort(key=lambda t: t[1], reverse=True)
        axes = [axis for axis, _ in strengths[: args.top_k_axes]]

    matrix = np.array([[run[axis] for axis in axes] for run in run_means], dtype=float)

    fig_w = max(16, len(axes) * 0.33)
    fig_h = max(12, len(run_labels) * 0.28)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    vmax = float(np.nanmax(np.abs(matrix))) if matrix.size else 1.0
    if vmax == 0.0:
        vmax = 1.0
    image = ax.imshow(matrix, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)

    ax.set_xticks(np.arange(len(axes)))
    ax.set_xticklabels(axes, rotation=90, fontsize=8)
    ax.set_yticks(np.arange(len(run_labels)))
    ax.set_yticklabels(run_labels, fontsize=9)
    ax.set_xlabel("Projection axis")
    ax.set_ylabel("User trait")
    metric_title = {"delta": "trait - neutral", "trait": "trait score", "neutral": "neutral score"}[args.metric]
    ax.set_title(args.title or f"User traits × axes heatmap ({metric_title})")

    cbar = fig.colorbar(image, ax=ax, fraction=0.02, pad=0.01)
    cbar.set_label(metric_title)

    plt.tight_layout()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=args.dpi)
    plt.close(fig)
    print(f"Saved heatmap to {output_path}")


if __name__ == "__main__":
    main()

