"""Exercise shipped backbone combinations with local, randomly initialized models."""

from pathlib import Path

import pytest
import torch
from PIL import Image
from transformers import (
    CLIPImageProcessor,
    CLIPVisionConfig,
    GemmaConfig,
    LlamaConfig,
    PhiConfig,
    Qwen2Config,
    SiglipImageProcessor,
    SiglipVisionConfig,
    StableLmConfig,
)

from tests.data.test_auto_vision_processing import tokenizer
from tinyllava.data.processor import TinyLlavaProcessor
from tinyllava.hparams.arguments import TuningArguments
from tinyllava.model import TinyLlavaConfig, TinyLlavaForConditionalGeneration
from tinyllava.model.llm.openelm import OpenELMConfig
from tinyllava.train.strategy.base import BaseTrainingStrategy
from tinyllava.train.workflow import training_stages

ROOT = Path(__file__).parents[2]
BACKBONES = {
    "gemma": GemmaConfig,
    "openelm": OpenELMConfig,
    "phi": PhiConfig,
    "qwen2_base": Qwen2Config,
    "qwen2_instruct": Qwen2Config,
    "stablelm": StableLmConfig,
    "tinyllama": LlamaConfig,
}


@pytest.mark.parametrize("preset", BACKBONES)
def test_preset_backbone_training_and_reload(tmp_path, preset):
    torch.set_num_threads(2)
    settings = training_stages(ROOT / f"configs/experiments/{preset}.yaml")[0][1][
        "model"
    ]
    text = BACKBONES[preset](
        vocab_size=4,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        # OpenELM owns variable-width layers and uses these native dimensions.
        **(
            {
                "model_dim": 16,
                "num_transformer_layers": 2,
                "num_gqa_groups": 1,
                "qkv_multipliers": 1,
                "ffn_multipliers": 2,
                "ffn_dim_divisor": 8,
            }
            if preset == "openelm"
            else {}
        ),
    )
    siglip = "siglip" in settings["vision_model_name_or_path"]
    vision = (SiglipVisionConfig if siglip else CLIPVisionConfig)(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        image_size=16,
        patch_size=8,
    )
    model = TinyLlavaForConditionalGeneration(
        TinyLlavaConfig(
            text_config=text,
            vision_config=vision,
            connector_config=settings["connector"],
            vision_feature_layer=settings["vision_feature_layer"],
            vision_feature_select_strategy=settings.get(
                "vision_feature_select_strategy"
            ),
        )
    )
    image_processor = (
        SiglipImageProcessor(size={"height": 16, "width": 16})
        if siglip
        else CLIPImageProcessor(
            size={"shortest_edge": 16}, crop_size={"height": 16, "width": 16}
        )
    )
    processor = TinyLlavaProcessor.from_model(
        tokenizer=tokenizer(),
        image_processor=image_processor,
        model=model,
    )
    assert processor.vision_feature_select_strategy == ("full" if siglip else "default")
    assert (
        processor.vision_feature_select_strategy
        == model.config.vision_feature_select_strategy
    )
    inputs = processor(
        text="hello <image> hello",
        images=Image.new("RGB", (16, 16), "red"),
        return_tensors="pt",
    )
    for policy in ("frozen", "full"):
        BaseTrainingStrategy(None, TuningArguments(tune_type_llm=policy))(model)
        model.gradient_checkpointing_enable()
        model.train()
        model.zero_grad(set_to_none=True)
        loss = model(**inputs, labels=inputs.input_ids).loss
        loss.backward()
        assert torch.isfinite(loss)
        assert model.model.multi_modal_projector.layers[0].weight.grad is not None
        assert not any(
            p.grad is not None for p in model.model.vision_tower.parameters()
        )
        assert (model.lm_head.weight.grad is not None) == (policy == "full")
    model.eval()
    model.save_pretrained(tmp_path)
    restored = TinyLlavaForConditionalGeneration.from_pretrained(tmp_path).eval()
    with torch.no_grad():
        torch.testing.assert_close(restored(**inputs).logits, model(**inputs).logits)
        assert (
            restored.generate(**inputs, max_new_tokens=2).shape[1]
            > inputs.input_ids.shape[1]
        )
