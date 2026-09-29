from pathlib import Path

import pytest

from tokenizers import Regex, Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Split
from transformers import PreTrainedTokenizerFast

from tinyllava.configuration import load_train_config, training_stages
from tinyllava.data.chat_template import inject_tinyllava_anchors
from tinyllava.data.chat_template.loading import resolve_chat_template

ROOT = Path(__file__).parents[2]
TEMPLATE = resolve_chat_template(
    chat_template_path=str(ROOT / "configs/chat_templates/llava.jinja")
)


def tokenizer():
    backend = Tokenizer(
        WordLevel(
            {
                "[UNK]": 0,
                "<|endoftext|>": 1,
                "<image>": 2,
                "Answer": 3,
                "Again": 4,
                " ": 5,
            },
            unk_token="[UNK]",
        )
    )
    backend.pre_tokenizer = Split(Regex(r"\w+|[^\w\s]|\s"), behavior="isolated")
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        eos_token="<|endoftext|>",
        additional_special_tokens=["<image>"],
        chat_template=inject_tinyllava_anchors(TEMPLATE).chat_template,
    )


def test_llava_preserves_questions_history_and_assistant_mask():
    tok = tokenizer()
    messages = [
        {"role": "system", "content": [{"type": "text", "text": "Be concise."}]},
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": "What is shown?"}],
        },
        {"role": "assistant", "content": [{"type": "text", "text": "Answer"}]},
        {"role": "user", "content": [{"type": "text", "text": "Explain more."}]},
        {"role": "assistant", "content": [{"type": "text", "text": "Again"}]},
    ]
    text = tok.apply_chat_template(messages, tokenize=False)
    assert text == (
        "Be concise. USER: <image>\nWhat is shown? ASSISTANT: Answer "
        "USER: Explain more. ASSISTANT: Again "
    )
    assert "<|endoftext|>" not in text
    assert tok.apply_chat_template(messages[1:], tokenize=False).startswith("USER:")
    encoded = tok.apply_chat_template(
        messages, tokenize=True, return_dict=True, return_assistant_tokens_mask=True
    )
    supervised = [
        i for i, mask in zip(encoded.input_ids, encoded.assistant_masks) if mask
    ]
    assert supervised == [3, 5, 4, 5]
    prompt = tok.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True
    )
    assert prompt.endswith("USER: Explain more. ASSISTANT:")


@pytest.mark.parametrize("preset", ["phi", "openelm"])
def test_presets_without_native_templates_use_llava(preset):
    for recipe in ("pretrain", "finetune"):
        config = load_train_config(
            ROOT / f"configs/train/{recipe}.yaml", [f"model={preset}"]
        )
        assert (
            config["model"]["chat_template_path"]
            == "configs/chat_templates/llava.jinja"
        )
    for path in (ROOT / "configs/experiments").glob(f"{preset}*.yaml"):
        for _, config, _ in training_stages(path):
            assert (
                config["model"]["chat_template_path"]
                == "configs/chat_templates/llava.jinja"
            )
