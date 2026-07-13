"""Bind sample-level image references to HF multimodal content blocks.

Legacy LLaVA-style samples keep image paths in top-level `image` or `images`
fields, while HF processors expect image payloads inside content blocks. This
module resolves path-like payloads and leaves actual loading/preprocessing to
the HF processor.
"""

import os
from collections.abc import Mapping, MutableMapping, Sequence
from typing import Any
from urllib.parse import urlparse

from transformers.utils.chat_template_utils import ChatType


def collect_sample_image_payloads(
    sample: Mapping[str, Any], image_folder: str | None = None
) -> list[Any]:
    """Collect sample-level image payloads and resolve relative paths."""
    images = sample.get("images", sample.get("image"))
    if images is None:
        return []
    image_payloads = [images] if isinstance(images, (str, os.PathLike)) else list(images)
    return [resolve_image_payload(image, image_folder=image_folder) for image in image_payloads]


def resolve_image_payload(image: Any, image_folder: str | None = None) -> Any:
    """Resolve path-like payloads and leave already-valid image objects untouched."""
    if not isinstance(image, (str, os.PathLike)):
        return image

    image_path = os.fspath(image)
    if os.path.isabs(image_path) or _is_remote_url(image_path) or image_folder is None:
        return image_path
    return os.path.join(image_folder, image_path)


def add_image_payloads(messages: ChatType, images: Sequence[Any]) -> None:
    """Attach collected image payloads to placeholder image blocks."""
    image_iter = iter(images)
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if isinstance(item, Mapping) and item.get("type") == "image" and not any(
                key in item for key in ("image", "url", "path", "base64")
            ):
                try:
                    item["image"] = next(image_iter)
                except StopIteration:
                    break


def resolve_message_image_payloads(
    messages: ChatType, image_folder: str | None = None
) -> None:
    """Resolve existing path-like payloads inside HF image content blocks."""
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, MutableMapping) or item.get("type") != "image":
                continue
            for key in ("image", "path"):
                if key in item:
                    item[key] = resolve_image_payload(item[key], image_folder=image_folder)


def _is_remote_url(path: str) -> bool:
    parsed = urlparse(path)
    return parsed.scheme in {"http", "https"}


__all__ = [
    "add_image_payloads",
    "collect_sample_image_payloads",
    "resolve_image_payload",
    "resolve_message_image_payloads",
]
