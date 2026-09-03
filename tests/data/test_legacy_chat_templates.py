from pathlib import Path

import pytest
from transformers.utils.chat_template_utils import _compile_jinja_template

from tinyllava.data.chat_template.loading import resolve_chat_template


TEMPLATE_DIR = Path(__file__).parents[2] / "configs" / "chat_templates"


def _messages(include_answer: bool = True):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "What is shown?"},
            ],
        }
    ]
    if include_answer:
        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "A cat"}],
            }
        )
    return messages


def test_pretrain_legacy_template_matches_paper_prompt():
    template = resolve_chat_template(
        chat_template_path=str(TEMPLATE_DIR / "pretrain_legacy.jinja")
    )
    rendered = _compile_jinja_template(template).render(
        messages=_messages(), add_generation_prompt=False
    )

    assert rendered == "<image>A cat\n"


def test_qwen2_base_legacy_template_matches_paper_prompt():
    template = resolve_chat_template(
        chat_template_path=str(TEMPLATE_DIR / "qwen2_base_legacy.jinja")
    )
    compiled = _compile_jinja_template(template)

    assert compiled.render(messages=_messages(), add_generation_prompt=False) == (
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's "
        "questions. USER: <image>\nWhat is shown? ASSISTANT: A cat<|endoftext|>"
    )
    assert compiled.render(
        messages=_messages(include_answer=False), add_generation_prompt=True
    ).endswith("USER: <image>\nWhat is shown? ASSISTANT:")


def test_chat_template_source_is_unambiguous(tmp_path):
    path = tmp_path / "template.jinja"
    path.write_text("from-file", encoding="utf-8")

    assert resolve_chat_template(chat_template_path=str(path)) == "from-file"
    assert resolve_chat_template(chat_template="inline") == "inline"
    with pytest.raises(ValueError, match="only one"):
        resolve_chat_template(chat_template="inline", chat_template_path=str(path))
