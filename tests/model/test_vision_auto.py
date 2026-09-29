import pytest
import torch
from transformers import (
    AutoConfig,
    AutoModel,
    CLIPConfig,
    CLIPVisionConfig,
    CLIPVisionModel,
    Dinov2Config,
    Dinov2Model,
    SiglipConfig,
    SiglipVisionConfig,
    SiglipVisionModel,
    ViTConfig,
    ViTModel,
)

from tinyllava.model import TinyLlavaConfig
from tinyllava.model.vision_tower import AutoVisionTowerModel


@pytest.mark.parametrize(
    "config_cls,model_cls",
    [
        (CLIPVisionConfig, CLIPVisionModel),
        (SiglipVisionConfig, SiglipVisionModel),
        (Dinov2Config, Dinov2Model),
        (ViTConfig, ViTModel),
    ],
)
def test_hf_vision_models_need_no_project_registry(tmp_path, config_cls, model_cls):
    vision = config_cls(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        image_size=16,
        patch_size=8,
    )
    config = TinyLlavaConfig(vision_config=vision.to_dict())
    model = AutoVisionTowerModel.from_config(config.vision_config).eval()
    assert isinstance(model, model_cls)
    model.save_pretrained(tmp_path)
    restored = AutoVisionTowerModel.from_pretrained(tmp_path).eval()
    pixels = torch.randn(1, 3, 16, 16)
    with torch.no_grad():
        expected = model(pixels, output_hidden_states=True)
        actual = restored(pixels, output_hidden_states=True)
    torch.testing.assert_close(actual.last_hidden_state, expected.last_hidden_state)


@pytest.mark.parametrize(
    "config_cls,vision_cls",
    [
        (CLIPConfig, CLIPVisionConfig),
        (SiglipConfig, SiglipVisionConfig),
    ],
)
@pytest.mark.parametrize("as_dict", [False, True])
def test_composite_vision_configs_are_normalized(config_cls, vision_cls, as_dict):
    source = config_cls()
    config = TinyLlavaConfig(vision_config=source.to_dict() if as_dict else source)
    assert isinstance(config.vision_config, vision_cls)


def test_project_auto_registration_reaches_tinyllava_config():
    from tinyllava.model.vision_tower import VISION_TOWER_CONFIG_MAPPING

    class CustomVisionConfig(CLIPVisionConfig):
        model_type = "test_custom_vision"

    class CustomVisionModel(CLIPVisionModel):
        config_class = CustomVisionConfig

    AutoConfig.register(CustomVisionConfig.model_type, CustomVisionConfig)
    AutoModel.register(CustomVisionConfig, CustomVisionModel)
    VISION_TOWER_CONFIG_MAPPING.register(
        CustomVisionConfig.model_type, CustomVisionConfig
    )
    AutoVisionTowerModel.register(CustomVisionConfig, CustomVisionModel)
    vision = CustomVisionConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        image_size=16,
        patch_size=8,
    )
    config = TinyLlavaConfig(vision_config=vision.to_dict())
    assert isinstance(config.vision_config, CustomVisionConfig)
    assert isinstance(
        AutoVisionTowerModel.from_config(config.vision_config), CustomVisionModel
    )


def test_mof_namespace_is_distinct_from_an_unqualified_hf_name(monkeypatch):
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    from tinyllava.model.vision_tower import VISION_TOWER_CONFIG_MAPPING
    from tinyllava.model.vision_tower.mof import MofVisionConfig

    class OtherMofConfig(CLIPVisionConfig):
        model_type = "mof"

    monkeypatch.setitem(CONFIG_MAPPING._extra_content, "mof", OtherMofConfig)
    assert AutoConfig.for_model("mof").__class__ is OtherMofConfig
    assert MofVisionConfig.model_type == "mof__tlf_vision_tower"
    assert VISION_TOWER_CONFIG_MAPPING[MofVisionConfig.model_type] is MofVisionConfig
    config = TinyLlavaConfig(vision_config=MofVisionConfig().to_dict())
    assert isinstance(config.vision_config, MofVisionConfig)
    assert AutoConfig.for_model("mof").__class__ is OtherMofConfig
