from transformers import AutoConfig, CLIPVisionConfig, LlamaConfig

from tinyllava.model.configuration_tinyllava import TinyLlavaConfig


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


def test_auto_config_round_trip_uses_tinyllava_registration(tmp_path):
    config = TinyLlavaConfig(
        text_config=LlamaConfig(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
        ),
        vision_config=CLIPVisionConfig(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            image_size=16,
            patch_size=8,
        ),
    )
    config.save_pretrained(tmp_path)

    restored = AutoConfig.from_pretrained(tmp_path)

    assert isinstance(restored, TinyLlavaConfig)
    assert restored.text_config.hidden_size == 16
    assert restored.vision_config.image_size == 16
