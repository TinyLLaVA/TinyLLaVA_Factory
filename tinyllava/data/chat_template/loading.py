"""Load an inline or file-backed chat template override."""

from __future__ import annotations

from pathlib import Path


def resolve_chat_template(
    *,
    chat_template: str | None = None,
    chat_template_path: str | None = None,
) -> str | None:
    """Resolve exactly one chat-template source."""

    if chat_template is not None and chat_template_path is not None:
        raise ValueError("Set only one of chat_template or chat_template_path.")
    if chat_template_path is None:
        return chat_template
    return Path(chat_template_path).expanduser().read_text(encoding="utf-8")


__all__ = ["resolve_chat_template"]
