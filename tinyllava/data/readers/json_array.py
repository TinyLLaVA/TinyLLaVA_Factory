"""Bounded-memory reader for legacy top-level JSON arrays."""

import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any


_READ_CHUNK_SIZE = 1 << 20


def is_json_array(path: Path) -> bool:
    """Return whether a local JSON file starts with a top-level array."""
    with path.open("rb") as stream:
        prefix = stream.read(4096)
    return prefix.lstrip(b"\xef\xbb\xbf \t\r\n").startswith(b"[")


def iter_json_array(path: str) -> Iterator[dict[str, Any]]:
    """Yield objects from a top-level JSON array without loading it all."""
    decoder = json.JSONDecoder()
    buffer = ""
    position = 0
    eof = False

    with open(path, encoding="utf-8-sig") as stream:
        def read_more() -> None:
            nonlocal buffer, position, eof
            chunk = stream.read(_READ_CHUNK_SIZE)
            buffer = buffer[position:] + chunk
            position = 0
            eof = not chunk

        def skip_whitespace() -> None:
            nonlocal position
            while True:
                while position < len(buffer) and buffer[position].isspace():
                    position += 1
                if position < len(buffer) or eof:
                    return
                read_more()

        read_more()
        skip_whitespace()
        if position >= len(buffer) or buffer[position] != "[":
            raise ValueError(f"Expected a top-level JSON array in {path!r}.")
        position += 1

        while True:
            skip_whitespace()
            if position < len(buffer) and buffer[position] == "]":
                position += 1
                break

            while True:
                try:
                    item, position = decoder.raw_decode(buffer, position)
                    break
                except json.JSONDecodeError:
                    if eof:
                        raise
                    read_more()

            if not isinstance(item, Mapping):
                raise TypeError(
                    "Training data must contain JSON objects, "
                    f"but found {type(item).__name__}."
                )
            yield dict(item)

            skip_whitespace()
            if position < len(buffer) and buffer[position] == ",":
                position += 1
                continue
            if position < len(buffer) and buffer[position] == "]":
                position += 1
                break
            if eof:
                raise ValueError(f"Expected ',' or ']' in {path!r}.")

        skip_whitespace()
        if position < len(buffer):
            raise ValueError(f"Unexpected content after the JSON array in {path!r}.")


def read_first_json_array_item(path: str) -> dict[str, Any]:
    """Read the first object for adapter auto-detection."""
    try:
        return next(iter_json_array(path))
    except StopIteration as exc:
        raise ValueError(f"Training data array {path!r} is empty.") from exc


__all__ = ["is_json_array", "iter_json_array", "read_first_json_array_item"]
