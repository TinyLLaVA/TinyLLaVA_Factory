import pytest
import torch
from transformers import (
    AutoConfig,
    AutoModel,
    CLIPVisionConfig,
    CLIPVisionModel,
    Dinov2Config,
    Dinov2Model,
    LlamaConfig,
)

from tinyllava.model import TinyLlavaConfig, TinyLlavaForConditionalGeneration
from tinyllava.model.connector.mof import MofConnector, MofConnectorConfig
from tinyllava.model.vision_tower.mof import MofVisionConfig, MofVisionModel


def tiny_vision_config():
    return MofVisionConfig(
        clip_config=CLIPVisionConfig(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            image_size=16,
            patch_size=8,
        ),
        dinov2_config=Dinov2Config(
            hidden_size=8,
            num_hidden_layers=2,
            num_attention_heads=2,
            image_size=16,
            patch_size=8,
        ),
    )


def tiny_config(**kwargs):
    kwargs.setdefault(
        "connector_config",
        {"model_type": "mof__tlf_connector", "vision_hidden_sizes": [16, 8]},
    )
    return TinyLlavaConfig(
        text_config=LlamaConfig(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
        ),
        vision_config=tiny_vision_config(),
        image_token_index=63,
        **kwargs,
    )


def test_mof_auto_round_trip_preserves_both_pretrained_branches(tmp_path):
    config = tiny_vision_config()
    clip = CLIPVisionModel(config.clip_config).eval()
    dino = Dinov2Model(config.dinov2_config).eval()
    clip.save_pretrained(tmp_path / "clip")
    dino.save_pretrained(tmp_path / "dino")
    model = MofVisionModel.from_pretrained_components(
        clip_model_name_or_path=tmp_path / "clip",
        dinov2_model_name_or_path=tmp_path / "dino",
    ).eval()
    for branch, original in ((model.clip, clip), (model.dinov2, dino)):
        for name, value in original.state_dict().items():
            torch.testing.assert_close(branch.state_dict()[name], value, rtol=0, atol=0)
    model.save_pretrained(tmp_path / "mof")
    restored_config = AutoConfig.from_pretrained(tmp_path / "mof")
    assert isinstance(restored_config, MofVisionConfig)
    assert restored_config.hidden_sizes == (16, 8)
    restored = AutoModel.from_pretrained(tmp_path / "mof").eval()
    pixels = torch.randn(1, 3, 16, 16)
    with torch.no_grad():
        actual = restored(pixels, output_hidden_states=True)
        clip_output = clip(pixels, output_hidden_states=True)
        dino_output = dino(pixels, output_hidden_states=True)
    assert actual.last_hidden_state.shape == (1, 5, 24)
    for packed, clip_state, dino_state in zip(
        actual.hidden_states,
        clip_output.hidden_states,
        dino_output.hidden_states,
        strict=True,
    ):
        torch.testing.assert_close(packed[..., :16], clip_state)
        torch.testing.assert_close(packed[..., 16:], dino_state)


def test_mof_connector_projects_branches_then_interleaves_multiple_layers():
    vision = tiny_vision_config()
    vision.clip_config.hidden_size = 2
    vision.dinov2_config.hidden_size = 1
    connector = MofConnector(
        MofConnectorConfig(depth=1, bias=False, vision_hidden_sizes=(2, 1)),
        vision_hidden_size=3,
        text_hidden_size=1,
        vision_feature_layer=[-2, -1],
    )
    with torch.no_grad():
        connector.clip.layers[0].weight.fill_(1)
        connector.dinov2.layers[0].weight.fill_(10)
    # Each selected layer contributes [CLIP channels, DINO channels].
    features = torch.tensor(
        [[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]]],
        requires_grad=True,
    )
    actual = connector(features)
    torch.testing.assert_close(
        actual, torch.tensor([[[12.0], [90.0], [36.0], [210.0]]])
    )
    actual.sum().backward()
    assert torch.all(features.grad != 0)
    assert connector.clip.layers[0].weight.grad is not None
    assert connector.dinov2.layers[0].weight.grad is not None


@pytest.mark.parametrize("strategy,tokens", [("default", 8), ("full", 10)])
def test_mof_composite_forward_backward_and_checkpoint_round_trip(
    tmp_path, strategy, tokens
):
    config = tiny_config(
        vision_feature_select_strategy=strategy, vision_feature_layer=[-2, -1]
    )
    assert config.connector_config.model_type == "mof__tlf_connector"
    model = TinyLlavaForConditionalGeneration(config).eval()
    ids = torch.tensor([[1] + [63] * tokens + [2]])
    pixels = torch.randn(1, 3, 16, 16)
    output = model(input_ids=ids, pixel_values=pixels, labels=ids)
    assert output.image_hidden_states.shape == (tokens, 16)
    features = model.model.get_image_features(
        pixels, image_sizes=torch.tensor([[16, 16]])
    )
    assert features.pooler_output[0].shape == (tokens, 16)
    output.loss.backward()
    assert (
        model.model.vision_tower.clip.embeddings.patch_embedding.weight.grad is not None
    )
    assert (
        model.model.vision_tower.dinov2.embeddings.patch_embeddings.projection.weight.grad
        is not None
    )
    model.save_pretrained(tmp_path)
    restored = TinyLlavaForConditionalGeneration.from_pretrained(tmp_path).eval()
    with torch.no_grad():
        restored_output = restored(input_ids=ids, pixel_values=pixels)
    torch.testing.assert_close(restored_output.logits, output.logits)


@pytest.mark.parametrize(
    "field,value", [("patch_size", 4), ("num_hidden_layers", 1), ("num_channels", 1)]
)
def test_mof_rejects_unaligned_branches(field, value):
    config = tiny_vision_config()
    setattr(config.dinov2_config, field, value)
    with pytest.raises(ValueError, match=field):
        MofVisionConfig(
            clip_config=config.clip_config, dinov2_config=config.dinov2_config
        )


def test_mof_connector_validates_its_own_input_contract():
    with pytest.raises(ValueError, match="must sum to vision_hidden_size"):
        MofConnector(
            MofConnectorConfig(),
            vision_hidden_size=768,
            text_hidden_size=16,
            vision_feature_layer=-2,
        )


def test_vision_selection_does_not_override_connector_selection():
    config = TinyLlavaConfig(vision_config=tiny_vision_config())
    assert config.connector_config.model_type == "mlp__tlf_connector"


@pytest.mark.parametrize("widths", [(0, 8), (16,), (-1, 8)])
def test_mof_connector_config_rejects_invalid_branch_widths(
    widths: tuple[int, ...],
) -> None:
    with pytest.raises(ValueError, match="two positive integers"):
        MofConnectorConfig(vision_hidden_sizes=widths)
