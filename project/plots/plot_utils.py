"""
Shared utilities for all plot_*.py scripts.

Kept intentionally minimal — only helpers that are genuinely reused across
multiple plot scripts live here. Per-script logic stays in the script itself.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

def aggregate(values: list[float], mode: str) -> float:
    """Return mean or median of values. Used when callers want to swap aggregation mode via CLI."""
    if not values:
        raise ValueError("Cannot aggregate empty list")
    if mode == "mean":
        return sum(values) / len(values)
    if mode == "median":
        values = sorted(values)
        n = len(values)
        mid = n // 2
        if n % 2 == 1:
            return values[mid]
        return (values[mid - 1] + values[mid]) / 2.0
    raise ValueError(f"Unknown aggregate mode: {mode}")


def infer_run_label(path: Path) -> str:
    """Infer a human-readable label from a projection output path.

    Newer outputs store the trait name in the JSONL rows directly. Older outputs
    encoded it in the directory structure as .../user_prompts/<trait>/..., so we
    fall back to extracting it from the path. If neither applies, the file stem is used.
    """
    parts = path.parts
    if "user_prompts" in parts:
        idx = parts.index("user_prompts")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return path.stem


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file fully into memory for plotting."""
    rows: list[dict[str, Any]] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_csv(rows: list[dict[str, Any]], path: Path, *, fieldnames: list[str]) -> None:
    """Write plot summary rows to CSV with explicit field order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


__all__ = ["aggregate", "infer_run_label", "load_jsonl", "write_csv"]
