import pytest
import torch
from transformers import CLIPVisionConfig, CLIPVisionModel

from tinyllava.model import TinyLlavaConfig, TinyLlavaForConditionalGeneration
from tinyllava.model.llm import AutoLanguageModelForCausalLM
from tinyllava.model.llm.openelm import OpenELMConfig, OpenELMForCausalLM


def tiny_config(**kwargs):
    return OpenELMConfig(
        vocab_size=32,
        model_dim=16,
        head_dim=4,
        num_transformer_layers=3,
        num_gqa_groups=2,
        qkv_multipliers=(0.5, 1),
        ffn_multipliers=(1, 2),
        ffn_dim_divisor=8,
        normalize_qk_projections=True,
        **kwargs,
    )


@pytest.mark.parametrize("shared", [True, False])
def test_openelm_round_trip_and_cached_decoding(tmp_path, shared):
    config = tiny_config(share_input_output_layers=shared)
    model = OpenELMForCausalLM(config).eval()
    assert (model.lm_head.weight is model.get_input_embeddings().weight) == shared
    ids = torch.tensor([[1, 4, 5, 6], [0, 1, 7, 8]])
    mask = ids.ne(0).long()
    positions = mask.cumsum(-1) - 1
    positions.masked_fill_(mask == 0, 0)
    with torch.no_grad():
        full = model(
            ids, attention_mask=mask, position_ids=positions, use_cache=False
        ).logits
        prefix = model(
            ids[:, :3],
            attention_mask=mask[:, :3],
            position_ids=positions[:, :3],
            use_cache=True,
        )
        step = model(
            ids[:, 3:],
            attention_mask=mask,
            position_ids=positions[:, 3:],
            past_key_values=prefix.past_key_values,
        )
    torch.testing.assert_close(step.logits[:, 0], full[:, -1], atol=1e-6, rtol=1e-5)
    assert step.past_key_values.get_seq_length() == 4
    model.save_pretrained(tmp_path)
    restored = AutoLanguageModelForCausalLM.from_pretrained(tmp_path).eval()
    with torch.no_grad():
        torch.testing.assert_close(
            restored(
                ids, attention_mask=mask, position_ids=positions, use_cache=False
            ).logits,
            full,
        )
    assert restored.config.hidden_size == 16
    assert restored.config.num_hidden_layers == 3
    generated = restored.generate(
        ids, attention_mask=mask, max_new_tokens=3, do_sample=False
    )
    generated_no_cache = restored.generate(
        ids, attention_mask=mask, max_new_tokens=3, do_sample=False, use_cache=False
    )
    torch.testing.assert_close(generated, generated_no_cache)


def test_openelm_multimodal_pretrained_components_and_training(tmp_path):
    lm = OpenELMForCausalLM(tiny_config(share_input_output_layers=True)).eval()
    vision_config = CLIPVisionConfig(
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        image_size=16,
        patch_size=8,
    )
    lm.save_pretrained(tmp_path / "lm")
    CLIPVisionModel(vision_config).save_pretrained(tmp_path / "vision")
    config = TinyLlavaConfig(
        text_config=lm.config, vision_config=vision_config, image_token_index=31
    )
    model = TinyLlavaForConditionalGeneration.from_pretrained_components(
        config,
        language_model_name_or_path=tmp_path / "lm",
        vision_model_name_or_path=tmp_path / "vision",
    ).eval()
    assert model.lm_head.weight is model.get_input_embeddings().weight
    torch.testing.assert_close(
        model.get_input_embeddings().weight, lm.get_input_embeddings().weight
    )
    model.gradient_checkpointing_enable()
    model.train()
    ids = torch.tensor([[1, 31, 31, 31, 31, 2]])
    pixels = torch.randn(1, 3, 16, 16)
    output = model(input_ids=ids, pixel_values=pixels, labels=ids)
    output.loss.backward()
    assert model.model.language_model.layers[0].attn.qkv_proj.weight.grad is not None
    assert model.model.multi_modal_projector.layers[0].weight.grad is not None
    model.eval()
    model.save_pretrained(tmp_path / "composite")
    restored = TinyLlavaForConditionalGeneration.from_pretrained(
        tmp_path / "composite"
    ).eval()
    with torch.no_grad():
        torch.testing.assert_close(
            restored(input_ids=ids, pixel_values=pixels).logits,
            model(input_ids=ids, pixel_values=pixels).logits,
        )
    assert (
        restored.generate(ids, pixel_values=pixels, max_new_tokens=2).shape[1]
        >= ids.shape[1] + 1
    )
