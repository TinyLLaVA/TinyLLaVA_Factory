"""Vision-side length rules compose with connectors and survive processor saves."""

from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from transformers import (
    CLIPImageProcessor,
    Dinov2WithRegistersConfig,
    Dinov2WithRegistersModel,
    DINOv3ViTConfig,
    DINOv3ViTModel,
    LlavaProcessor,
    PreTrainedConfig,
)

from transformers.image_processing_utils import BatchFeature

from tests.data.test_auto_vision_processing import tokenizer
from tinyllava.data.processor import AutoProcessor, TinyLlavaProcessor
from tinyllava.model import TinyLlavaConfig
from tinyllava.model.connector import CONNECTOR_CONFIG_MAPPING
from tinyllava.model.vision_tower import VISION_TOWER_CONFIG_MAPPING


class GridVisionConfig(PreTrainedConfig):
    """Test extension using processed grid metadata, without a pixel-size formula."""

    model_type = "test_grid__tlf_vision_tower"
    num_additional_image_tokens = 0

    def get_output_sequence_length(
        self,
        image_inputs: dict[str, torch.Tensor],
        image_idx: int,
        *,
        vision_feature_select_strategy: str,
    ) -> int:
        if vision_feature_select_strategy != "full":
            raise ValueError("GridVisionConfig expects full patch selection")
        return int(image_inputs["image_grid_thw"][image_idx].prod())


def small_image_processor():
    return CLIPImageProcessor(
        size={"shortest_edge": 16}, crop_size={"height": 16, "width": 16}
    )


@pytest.mark.parametrize(
    "connector_type,expected",
    [("mlp", [6, 12]), ("mof", [12, 24]), ("resampler", [3, 3])],
)
def test_vision_hook_bypasses_llava_and_composes_with_connector(
    tmp_path, monkeypatch, connector_type, expected
):
    monkeypatch.setitem(
        VISION_TOWER_CONFIG_MAPPING._extra_content,
        GridVisionConfig.model_type,
        GridVisionConfig,
    )
    connector = CONNECTOR_CONFIG_MAPPING[connector_type + "__tlf_connector"](
        **({"num_queries": 3} if connector_type == "resampler" else {})
    )
    config = TinyLlavaConfig(
        vision_config=GridVisionConfig(), connector_config=connector
    )
    processor = TinyLlavaProcessor.from_model(
        tokenizer=tokenizer(),
        image_processor=small_image_processor(),
        model=SimpleNamespace(config=config),
    )

    def unexpected_fallback(*args, **kwargs):
        pytest.fail("Custom vision lengths must bypass HF LLaVA's image-size rule")

    monkeypatch.setattr(LlavaProcessor, "replace_image_token", unexpected_fallback)
    # No pixel_values: the VT, rather than TinyLlavaProcessor, owns required metadata.
    image_inputs = {"image_grid_thw": torch.tensor([[1, 2, 3], [1, 3, 4]])}
    for idx, count in enumerate(expected):
        assert processor.replace_image_token(image_inputs, idx) == "<image>" * count

    processor.save_pretrained(tmp_path)
    restored = AutoProcessor.from_pretrained(tmp_path, local_files_only=True)
    assert isinstance(restored.vision_config, GridVisionConfig)
    for idx, count in enumerate(expected):
        assert restored.replace_image_token(image_inputs, idx) == "<image>" * count

    # The unused size-only API must not silently return the LLaVA estimate.
    with pytest.raises(NotImplementedError, match="Size-only token estimation"):
        restored._get_num_multimodal_tokens(image_sizes=[(16, 16)])
    assert restored._get_num_multimodal_tokens().num_image_tokens is None

    # A hook error is an extension error, not permission to fall back to HF.
    restored.vision_feature_select_strategy = "default"
    with pytest.raises(ValueError, match="expects full patch selection"):
        restored.replace_image_token(image_inputs, 0)


@pytest.mark.parametrize(
    "config_cls,model_cls",
    [
        (Dinov2WithRegistersConfig, Dinov2WithRegistersModel),
        (DINOv3ViTConfig, DINOv3ViTModel),
    ],
)
@pytest.mark.parametrize("strategy", [None, "default", "full"])
def test_native_register_tokens_match_actual_vision_features(
    tmp_path, config_cls, model_cls, strategy
):
    vision_config = config_cls(
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        patch_size=8,
        num_register_tokens=2,
    )
    config = TinyLlavaConfig(
        vision_config=vision_config, vision_feature_select_strategy=strategy
    )
    processor = TinyLlavaProcessor.from_model(
        tokenizer=tokenizer(),
        image_processor=small_image_processor(),
        model=SimpleNamespace(config=config),
    )
    inputs = processor(
        text="<image>", images=Image.new("RGB", (16, 16)), return_tensors="pt"
    )
    with torch.no_grad():
        features = model_cls(vision_config).eval()(inputs.pixel_values).last_hidden_state
    if config.vision_feature_select_strategy == "default":
        features = features[:, 1:]
    expected = features.shape[1]
    assert expected == (7 if strategy == "full" else 6)
    assert inputs.input_ids.eq(processor.image_token_id).sum().item() == expected
    assert processor._get_num_multimodal_tokens(
        image_sizes=[(16, 16)]
    ).num_image_tokens == [expected]

    processor.save_pretrained(tmp_path)
    restored = AutoProcessor.from_pretrained(tmp_path, local_files_only=True)
    assert type(restored.vision_config) is config_cls
    assert restored.num_additional_image_tokens == 3
    torch.testing.assert_close(
        restored(text="<image>", images=Image.new("RGB", (16, 16)), return_tensors="pt").input_ids,
        inputs.input_ids,
    )


def test_custom_vision_length_is_used_by_full_processor_call(monkeypatch):
    processor = TinyLlavaProcessor.from_model(
        tokenizer=tokenizer(),
        image_processor=small_image_processor(),
        model=SimpleNamespace(
            config=TinyLlavaConfig(
                vision_config=GridVisionConfig(),
                connector_config={"model_type": "mof__tlf_connector"},
            )
        ),
    )

    def preprocess(images, **kwargs):
        # A non-raster output makes any accidental LLaVA fallback fail.
        return BatchFeature(
            {"image_grid_thw": [[1, 2, 3], [1, 3, 4]]}, tensor_type="pt"
        )

    monkeypatch.setattr(processor.image_processor, "preprocess", preprocess)
    result = processor(
        text=["<image>", "<image>"],
        images=[Image.new("RGB", (16, 24)), Image.new("RGB", (24, 32))],
        padding=True,
        return_tensors="pt",
    )
    assert result.input_ids.eq(processor.image_token_id).sum(dim=1).tolist() == [12, 24]
    assert result.image_grid_thw.tolist() == [[1, 2, 3], [1, 3, 4]]
