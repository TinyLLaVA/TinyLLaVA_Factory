"""Load an inline or file-backed chat template override."""

from __future__ import annotations

from pathlib import Path


def resolve_chat_template(
    *,
    chat_template: str | None = None,
    chat_template_path: str | None = None,
) -> str | None:
    """Read a Jinja chat template from an inline string or a UTF-8 file.

    Args:
        chat_template: Inline template, or `None` to use a file or the processor default.
        chat_template_path: Template file path; `~` is expanded.

    Returns:
        The template text, or `None` when neither source is supplied.

    Raises:
        ValueError: Both template sources are supplied.
        OSError: The template file cannot be read.
    """

    if chat_template is not None and chat_template_path is not None:
        raise ValueError("Set only one of chat_template or chat_template_path.")
    if chat_template_path is None:
        return chat_template
    return Path(chat_template_path).expanduser().read_text(encoding="utf-8")


__all__ = ["resolve_chat_template"]
