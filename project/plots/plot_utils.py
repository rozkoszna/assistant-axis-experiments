from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

def aggregate(values: list[float], mode: str) -> float:
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
    parts = path.parts
    if "user_prompts" in parts:
        idx = parts.index("user_prompts")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return path.stem


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_csv(rows: list[dict[str, Any]], path: Path, *, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


__all__ = ["aggregate", "infer_run_label", "load_jsonl", "write_csv"]
