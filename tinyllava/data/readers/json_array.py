"""Incremental reader for top-level JSON arrays of training objects."""

from collections.abc import Iterator, Mapping
from contextlib import closing
from decimal import Decimal
from pathlib import Path
from typing import Any, BinaryIO

from ijson import JSONError
from ijson.backends import yajl2_c as ijson


# The Python backend accepts non-JSON Unicode whitespace; require YAJL.
_READ_CHUNK_SIZE = 1 << 16
_JSON_WHITESPACE = b" \t\r\n"


def _seek_array_start(stream: BinaryIO) -> bool:
    """Skip an optional UTF-8 BOM and JSON whitespace in a local file."""
    if stream.read(3) != b"\xef\xbb\xbf":
        stream.seek(0)
    while chunk := stream.read(_READ_CHUNK_SIZE):
        content = chunk.lstrip(_JSON_WHITESPACE)
        if content:
            stream.seek(-len(content), 1)
            return content.startswith(b"[")
    return False


def is_json_array(path: Path) -> bool:
    """Return whether a local JSON file starts with a top-level array."""
    with path.open("rb") as stream:
        return _seek_array_start(stream)


def iter_json_array(path: str) -> Iterator[dict[str, Any]]:
    """Yield objects with memory proportional to the buffer and largest row.

    Accept an optional UTF-8 BOM, but require standard JSON syntax. Numbers
    retain json.loads semantics: arbitrary-size integers and float decimals.
    The complete document is validated when the iterator is exhausted.
    """
    with open(path, "rb") as stream:
        if not _seek_array_start(stream):
            raise ValueError(f"Expected a top-level JSON array in {path!r}.")
        try:
            # use_float=True also limits integer range in some ijson backends.
            # Convert only Decimal events to preserve arbitrary-size integers.
            events = (
                (prefix, event, float(value) if isinstance(value, Decimal) else value)
                for prefix, event, value in ijson.parse(
                    stream, buf_size=_READ_CHUNK_SIZE
                )
            )
            for item in ijson.items(events, "item"):
                if not isinstance(item, Mapping):
                    raise TypeError(
                        "Training data must contain JSON objects, "
                        f"but found {type(item).__name__}."
                    )
                yield dict(item)
        except JSONError as exc:
            raise ValueError(f"Invalid JSON array in {path!r}: {exc}") from exc


def read_first_json_array_item(path: str) -> dict[str, Any]:
    """Read the first object for adapter auto-detection, then close the file."""
    with closing(iter_json_array(path)) as items:
        try:
            return next(items)
        except StopIteration as exc:
            raise ValueError(f"Training data array {path!r} is empty.") from exc


__all__ = ["is_json_array", "iter_json_array", "read_first_json_array_item"]
