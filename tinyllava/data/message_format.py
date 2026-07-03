"""Convert legacy LLaVA-style samples into Hugging Face chat messages.

The target legacy shape is used by datasets such as
`liuhaotian/LLaVA-Pretrain`:

    {
        "id": "002239345",
        "image": "00223/002239345.jpg",
        "conversations": [
            {
                "from": "human",
                "value": "Write a terse but informative summary of the picture.\n<image>"
            },
            {
                "from": "gpt",
                "value": "a grey watch with an army style strap"
            }
        ]
    }

This module only normalizes message schema. It does not tokenize text, render
chat templates, load images, or construct labels.
"""

from collections.abc import Mapping, Sequence
from typing import Any

from transformers.utils.chat_template_utils import ChatType

from tinyllava.utils.constants import DEFAULT_IMAGE_TOKEN

ROLE_MAP = {
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "system": "system",
}


def normalize_messages(sample: Mapping[str, Any]) -> ChatType:
    """Convert one training sample into HF-style chat messages."""
    messages = sample.get("messages")
    if messages is None:
        messages = sample.get("conversations")
    if messages is None:
        raise KeyError("Training sample must contain either `messages` or `conversations`.")

    return [_normalize_message(message) for message in messages]


def _normalize_message(message: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize one message role/content pair."""
    role = message.get("role", message.get("from"))
    if role is None:
        raise KeyError("Conversation message must contain either `role` or `from`.")

    content = message.get("content", message.get("value", ""))
    return {
        "role": ROLE_MAP.get(str(role), str(role)),
        "content": normalize_content(content),
    }


def normalize_content(content: Any) -> list[dict[str, Any]]:
    """Normalize content into HF multimodal content blocks."""
    if isinstance(content, str):
        return _split_legacy_image_markers(content)
    if isinstance(content, Sequence):
        normalized = []
        for item in content:
            if isinstance(item, str):
                normalized.extend(_split_legacy_image_markers(item))
            elif isinstance(item, Mapping):
                normalized.append(dict(item))
            else:
                normalized.append({"type": "text", "text": str(item)})
        return normalized
    return [{"type": "text", "text": str(content)}]


def _split_legacy_image_markers(text: str) -> list[dict[str, Any]]:
    """Split LLaVA's inline `<image>` marker into image/text content blocks."""
    parts = text.split(DEFAULT_IMAGE_TOKEN)
    content: list[dict[str, Any]] = []
    for index, part in enumerate(parts):
        if index > 0:
            content.append({"type": "image"})
        if part:
            content.append({"type": "text", "text": part.lstrip("\n") if index > 0 else part})
    return content or [{"type": "text", "text": ""}]


__all__ = ["normalize_content", "normalize_messages"]
