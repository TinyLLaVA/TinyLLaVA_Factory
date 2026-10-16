import json

import pytest
from transformers import (
    AutoConfig,
    CLIPVisionConfig,
    Dinov2WithRegistersConfig,
    LlamaConfig,
    SiglipConfig,
    SiglipVisionConfig,
    Siglip2VisionConfig,
)

from tinyllava.model.configuration_tinyllava import TinyLlavaConfig
from tinyllava.model.vision_tower.mof import MofVisionConfig


def test_default_config_uses_hf_nested_subconfigs():
    config = TinyLlavaConfig()

    assert isinstance(config.text_config, LlamaConfig)
    assert isinstance(config.vision_config, CLIPVisionConfig)
    assert config.connector_config.model_type == "mlp__tlf_connector"
    assert config.vision_feature_layer == -2
    assert config.vision_feature_select_strategy == "default"


def test_config_builds_typed_subconfigs_from_dicts():
    config = TinyLlavaConfig(
        text_config={
            "model_type": "llama",
            "vocab_size": 128,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
        },
        vision_config={
            "model_type": "clip_vision_model",
            "hidden_size": 48,
            "intermediate_size": 96,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "image_size": 32,
            "patch_size": 16,
        },
        connector_config={
            "model_type": "mlp__tlf_connector",
            "depth": 1,
            "act": "relu",
            "bias": False,
        },
        vision_feature_layer=-1,
        vision_feature_select_strategy="full",
    )

    assert isinstance(config.text_config, LlamaConfig)
    assert config.text_config.hidden_size == 32
    assert isinstance(config.vision_config, CLIPVisionConfig)
    assert config.vision_config.hidden_size == 48
    assert config.connector_config.depth == 1
    assert config.connector_config.act == "relu"
    assert config.connector_config.bias is False
    assert config.vision_feature_layer == -1
    assert config.vision_feature_select_strategy == "full"


@pytest.mark.parametrize(
    "vision_config,strategy,extra_tokens",
    [
        (CLIPVisionConfig(), "default", 1),
        (SiglipVisionConfig().to_dict(), "full", 0),
        (SiglipConfig(), "full", 0),
        (SiglipConfig().to_dict(), "full", 0),
        (Siglip2VisionConfig(), "full", 0),
        (Dinov2WithRegistersConfig(num_register_tokens=2), "default", 3),
        (MofVisionConfig(), "default", 1),
    ],
    ids=[
        "clip",
        "siglip-dict",
        "siglip-composite",
        "siglip-composite-dict",
        "siglip2",
        "dino-registers",
        "mof",
    ],
)
def test_inferred_selection_survives_auto_config_round_trip(
    tmp_path, vision_config, strategy, extra_tokens
):
    config = TinyLlavaConfig(
        text_config=LlamaConfig(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
        ),
        vision_config=vision_config,
    )
    assert config.vision_feature_select_strategy == strategy
    assert config.num_additional_image_tokens == extra_tokens
    config.save_pretrained(tmp_path)
    saved = json.loads((tmp_path / "config.json").read_text())
    assert saved["vision_feature_select_strategy"] == strategy

    restored = AutoConfig.from_pretrained(tmp_path)

    assert isinstance(restored, TinyLlavaConfig)
    assert restored.text_config.hidden_size == 16
    assert type(restored.vision_config) is type(config.vision_config)
    assert restored.vision_feature_select_strategy == strategy
    assert restored.num_additional_image_tokens == extra_tokens
