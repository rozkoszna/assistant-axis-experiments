#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create one heatmap per axis showing topic-wise movement "
            "(trait mean - neutral mean) for user traits."
        )
    )
    parser.add_argument("--per-topic-csv", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument(
        "--axes",
        nargs="*",
        default=None,
        help="Optional subset of axes. Default: all axes in CSV.",
    )
    parser.add_argument(
        "--top-traits",
        type=int,
        default=20,
        help="Keep top N traits per axis by mean absolute movement.",
    )
    return parser.parse_args()


def safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in name).strip("_")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.per_topic_csv, newline="") as f:
        rows = list(csv.DictReader(f))

    by_axis_topic_neutral: dict[str, dict[str, float]] = {}
    by_axis_trait_topic: dict[str, dict[str, dict[str, float]]] = {}
    axes_seen: set[str] = set()
    topics_seen: set[str] = set()

    for row in rows:
        axis = row["axis"]
        topic = row["topic"]
        condition = row["condition"]
        value = float(row["mean"])
        axes_seen.add(axis)
        topics_seen.add(topic)

        if condition == "Neutral":
            by_axis_topic_neutral.setdefault(axis, {})[topic] = value
            continue
        by_axis_trait_topic.setdefault(axis, {}).setdefault(condition, {})[topic] = value

    selected_axes = sorted(args.axes) if args.axes else sorted(axes_seen)
    topics = sorted(topics_seen, key=lambda x: int(x) if x.isdigit() else x)

    for axis in selected_axes:
        neutral_by_topic = by_axis_topic_neutral.get(axis, {})
        trait_map = by_axis_trait_topic.get(axis, {})
        if not neutral_by_topic or not trait_map:
            continue

        movement_by_trait: dict[str, dict[str, float]] = {}
        for trait, trait_topic_map in trait_map.items():
            common_topics = [t for t in topics if t in neutral_by_topic and t in trait_topic_map]
            if not common_topics:
                continue
            movement_by_trait[trait] = {
                t: trait_topic_map[t] - neutral_by_topic[t] for t in common_topics
            }

        if not movement_by_trait:
            continue

        trait_rank = sorted(
            movement_by_trait.keys(),
            key=lambda trait: np.mean([abs(v) for v in movement_by_trait[trait].values()]),
            reverse=True,
        )
        if args.top_traits > 0:
            trait_rank = trait_rank[: args.top_traits]

        matrix = np.full((len(trait_rank), len(topics)), np.nan, dtype=float)
        for i, trait in enumerate(trait_rank):
            for j, topic in enumerate(topics):
                if topic in movement_by_trait[trait]:
                    matrix[i, j] = movement_by_trait[trait][topic]

        vmax = np.nanmax(np.abs(matrix))
        if not np.isfinite(vmax) or vmax == 0:
            vmax = 1.0

        fig, ax = plt.subplots(figsize=(max(8, 0.8 * len(topics)), max(6, 0.35 * len(trait_rank))))
        im = ax.imshow(matrix, cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(topics)))
        ax.set_xticklabels(topics)
        ax.set_yticks(range(len(trait_rank)))
        ax.set_yticklabels(trait_rank)
        ax.set_xlabel("Topic (intent_index)")
        ax.set_ylabel("User trait")
        ax.set_title(f"Axis: {axis} | movement vs neutral by topic")
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Trait mean - Neutral mean")
        fig.tight_layout()

        out_path = output_dir / f"{safe_name(axis)}__trait_topic_movement.png"
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()

