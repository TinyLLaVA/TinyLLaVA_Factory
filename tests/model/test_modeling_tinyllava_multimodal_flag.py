import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import warnings

import pytest
import torch
from transformers import CLIPVisionConfig, CLIPVisionModel, LlamaConfig, LlamaForCausalLM

from tinyllava.model.configuration_tinyllava import TinyLlavaConfig
from tinyllava.model.modeling_tinyllava import TinyLlavaForConditionalGeneration


def _write_json(path: Path, content: dict) -> None:
    path.write_text(json.dumps(content), encoding="utf-8")


def _build_tiny_tokenizer(tokenizer_dir: Path) -> None:
    vocab_file = tokenizer_dir / "vocab.txt"
    vocab_file.write_text(
        "\n".join(
            [
                "[PAD]",
                "[UNK]",
                "[CLS]",
                "[SEP]",
                "[MASK]",
                "hello",
                "world",
                "image",
                "token",
            ]
        ),
        encoding="utf-8",
    )

    _write_json(
        tokenizer_dir / "tokenizer_config.json",
        {
            "tokenizer_class": "BertTokenizerFast",
            "do_lower_case": False,
            "unk_token": "[UNK]",
            "pad_token": "[PAD]",
            "cls_token": "[CLS]",
            "sep_token": "[SEP]",
            "mask_token": "[MASK]",
        },
    )
    _write_json(
        tokenizer_dir / "special_tokens_map.json",
        {
            "unk_token": "[UNK]",
            "pad_token": "[PAD]",
            "cls_token": "[CLS]",
            "sep_token": "[SEP]",
            "mask_token": "[MASK]",
        },
    )
    _write_json(tokenizer_dir / "config.json", {"model_type": "bert"})


def _build_tiny_llama_checkpoint(llm_dir: Path) -> None:
    config = LlamaConfig.from_dict(
        {
            "vocab_size": 128,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "max_position_embeddings": 64,
            "bos_token_id": 2,
            "eos_token_id": 3,
            "pad_token_id": 1,
        }
    )
    model = LlamaForCausalLM(config)
    model.save_pretrained(llm_dir)


def _build_tiny_clip_checkpoint(vision_dir: Path) -> None:
    vision_config = CLIPVisionConfig.from_dict(
        {
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "image_size": 32,
            "patch_size": 16,
            "projection_dim": 32,
        }
    )
    vision_model = CLIPVisionModel(vision_config)
    vision_model.save_pretrained(vision_dir)

    _write_json(
        vision_dir / "preprocessor_config.json",
        {
            "do_resize": True,
            "size": {"shortest_edge": 32},
            "do_center_crop": True,
            "crop_size": {"height": 32, "width": 32},
            "do_normalize": True,
            "image_mean": [0.5, 0.5, 0.5],
            "image_std": [0.5, 0.5, 0.5],
            "resample": 3,
        },
    )


@pytest.fixture
def tiny_tinyllava_model(tmp_path: Path) -> TinyLlavaForConditionalGeneration:
    llm_dir = tmp_path / "tinyllama-local"
    llm_dir.mkdir()
    _build_tiny_llama_checkpoint(llm_dir)

    tokenizer_dir = tmp_path / "tiny-tokenizer"
    tokenizer_dir.mkdir()
    _build_tiny_tokenizer(tokenizer_dir)

    vision_dir = tmp_path / "clip-local"
    vision_dir.mkdir()
    _build_tiny_clip_checkpoint(vision_dir)

    config = TinyLlavaConfig()
    config.load_from_config(
        SimpleNamespace(
            model_name_or_path=str(llm_dir),
            tokenizer_name_or_path=str(tokenizer_dir),
            vision_tower=f"clip:{vision_dir}",
            vision_tower2="",
            connector_type="linear",
            mm_vision_select_layer=-2,
            mm_vision_select_feature="patch",
            image_aspect_ratio="square",
            model_max_length=64,
            tokenizer_padding_side="right",
            tokenizer_use_fast=True,
            cache_dir=None,
            resampler_hidden_size=None,
            num_queries=None,
            num_resampler_layers=None,
        )
    )
    model = TinyLlavaForConditionalGeneration(config)
    # Ensure the test walks the actual HF `from_pretrained` loading path.
    model.load_llm(model_name_or_path=str(llm_dir))
    model.load_vision_tower(model_name_or_path=str(vision_dir))
    model.eval()
    return model


def _collect_messages(caught_warnings: list[warnings.WarningMessage]) -> list[str]:
    return [str(item.message) for item in caught_warnings]


def test_forward_warns_and_skips_multimodal_when_disabled_with_images(
    tiny_tinyllava_model: TinyLlavaForConditionalGeneration,
):
    model = tiny_tinyllava_model
    input_ids = torch.tensor([[2, 4, 5, 3]], dtype=torch.long)
    images = torch.zeros((1, 3, 32, 32), dtype=torch.float32)

    with patch.object(
        model,
        "prepare_inputs_labels_for_multimodal",
        side_effect=AssertionError("multimodal path should be skipped"),
    ):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    images=images,
                    is_multimodal=False,
                    return_dict=True,
                )

    messages = _collect_messages(caught)
    assert any("`images` is provided but `is_multimodal=False`" in m for m in messages)
    assert outputs.logits.shape[:2] == input_ids.shape


def test_forward_warns_when_enabled_but_images_missing(
    tiny_tinyllava_model: TinyLlavaForConditionalGeneration,
):
    model = tiny_tinyllava_model
    input_ids = torch.tensor([[2, 4, 5, 3]], dtype=torch.long)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                images=None,
                is_multimodal=True,
                return_dict=True,
            )

    messages = _collect_messages(caught)
    assert any("`is_multimodal=True` but `images` is None" in m for m in messages)
    assert outputs.logits.shape[:2] == input_ids.shape


def test_generate_warns_and_skips_multimodal_when_disabled_with_images(
    tiny_tinyllava_model: TinyLlavaForConditionalGeneration,
):
    model = tiny_tinyllava_model
    input_ids = torch.tensor([[2, 4, 5, 3]], dtype=torch.long)
    images = torch.zeros((1, 3, 32, 32), dtype=torch.float32)
    max_new_tokens = 4

    with patch.object(
        model,
        "prepare_inputs_labels_for_multimodal",
        side_effect=AssertionError("multimodal path should be skipped"),
    ):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with torch.no_grad():
                generated = model.generate(
                    inputs=input_ids,
                    images=images,
                    is_multimodal=False,
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                )

    messages = _collect_messages(caught)
    assert any("`images` is provided but `is_multimodal=False`" in m for m in messages)
    assert isinstance(generated, torch.Tensor)
    assert generated.shape[0] == input_ids.shape[0]
    assert generated.shape[1] == max_new_tokens


def test_generate_warns_when_enabled_but_images_missing(
    tiny_tinyllava_model: TinyLlavaForConditionalGeneration,
):
    model = tiny_tinyllava_model
    input_ids = torch.tensor([[2, 4, 5, 3]], dtype=torch.long)
    max_new_tokens = 5

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with torch.no_grad():
            generated = model.generate(
                inputs=input_ids,
                images=None,
                is_multimodal=True,
                do_sample=False,
                max_new_tokens=max_new_tokens,
            )

    messages = _collect_messages(caught)
    assert any("`is_multimodal=True` but `images` is None" in m for m in messages)
    assert isinstance(generated, torch.Tensor)
    assert generated.shape[0] == input_ids.shape[0]
    assert generated.shape[1] == max_new_tokens
