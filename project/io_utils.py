from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Iterator


# ------------------------
# JSONL helpers
# ------------------------

def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """
    Stream JSONL file row by row.

    Yields:
        dict per line
    """
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """
    Load entire JSONL file into memory.
    """
    return list(read_jsonl(path))


def write_jsonl(rows: Iterable[dict[str, Any]], path: Path) -> None:
    """
    Write iterable of dicts to JSONL file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def append_jsonl(rows: Iterable[dict[str, Any]], path: Path) -> None:
    """
    Append rows to an existing JSONL file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


# ------------------------
# JSON helpers
# ------------------------

def load_json(path: Path) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def write_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ------------------------
# Trait helpers
# ------------------------

def load_trait_list(path: Path) -> list[str]:
    """
    Load a JSON file containing a list of trait names.

    Example:
        ["confused", "supportive", "concise"]
    """
    data = load_json(path)

    if not isinstance(data, list) or not all(isinstance(x, str) for x in data):
        raise ValueError(f"{path} must contain a JSON list of strings")

    return data


# ------------------------
# Run helpers
# ------------------------

def make_run_name(user_value: str | None = None) -> str:
    """
    Generate a run name if not provided.
    """
    if user_value:
        return user_value
    return datetime.now().strftime("run_%Y%m%d_%H%M%S")


# ------------------------
# Debug / logging helpers
# ------------------------

def print_header(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def print_kv(key: str, value: Any) -> None:
    print(f"{key}: {value}")