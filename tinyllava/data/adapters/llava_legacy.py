"""Adapter for legacy LLaVA conversation JSON files."""

from collections.abc import Mapping, Sequence
from typing import Any

from datasets import Features, List, Value

from tinyllava.data.message_format import ROLE_MAP


class LlavaLegacyDatasetAdapter:
    """Drop source metadata and canonicalize `from`/`value` conversations."""

    name = "llava_legacy"
    cache_version = "1"
    features = Features(
        {
            "id": Value("string"),
            "image": Value("string"),
            "messages": List(
                {
                    "role": Value("string"),
                    "content": Value("string"),
                }
            ),
        }
    )

    def adapt(self, sample: Mapping[str, Any]) -> dict[str, Any]:
        conversations = sample.get("conversations")
        if not isinstance(conversations, Sequence) or isinstance(conversations, str):
            raise TypeError("LLaVA legacy samples must contain a `conversations` list.")

        messages = []
        for index, message in enumerate(conversations):
            if not isinstance(message, Mapping):
                raise TypeError(f"Conversation item {index} must be an object.")
            role = message.get("from", message.get("role"))
            if role is None:
                raise KeyError(f"Conversation item {index} has no `from` or `role`.")
            content = message.get("value", message.get("content", ""))
            if not isinstance(content, str):
                raise TypeError(
                    f"Conversation item {index} content must be text, "
                    f"but found {type(content).__name__}."
                )
            role = str(role)
            messages.append(
                {
                    "role": ROLE_MAP.get(role, role),
                    "content": content,
                }
            )

        image = sample.get("image")
        if image is not None and not isinstance(image, str):
            raise TypeError("LLaVA legacy `image` must be a string path or null.")
        return {
            "id": str(sample.get("id", "")),
            "image": image,
            "messages": messages,
        }


__all__ = ["LlavaLegacyDatasetAdapter"]
