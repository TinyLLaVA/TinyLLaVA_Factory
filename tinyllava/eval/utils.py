"""Shared utilities for evaluation datasets and result files."""

from __future__ import annotations

import json
import os
from typing import Any


def read_jsonl(path: str) -> list[dict[str, Any]]:
    with open(os.path.expanduser(path), encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def read_jsonl_with_errors(path: str) -> tuple[list[dict[str, Any]], int]:
    rows = []
    error_lines = 0
    with open(os.path.expanduser(path), encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                error_lines += 1
    return rows, error_lines


def write_json(path: str, data: Any, *, indent: int | None = None) -> None:
    output_path = os.path.expanduser(path)
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent)
