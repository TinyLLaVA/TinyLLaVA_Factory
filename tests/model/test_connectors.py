import subprocess
import sys

import pytest
import torch
from transformers import CLIPVisionConfig, LlamaConfig

from tinyllava.model import TinyLlavaConfig, TinyLlavaForConditionalGeneration
from tinyllava.model.connector import CONNECTOR_CONFIG_MAPPING, AutoConnectorModel


def connector_config(name):
    kwargs = (
        {}
        if name == "identity"
        else dict(
            num_queries=3,
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
        )
    )
    return CONNECTOR_CONFIG_MAPPING[name + "__tlf_connector"](**kwargs)


@pytest.mark.parametrize("name", ["identity", "qformer", "resampler"])
def test_connector_multimodal_training_and_round_trip(tmp_path, name):
    config = TinyLlavaConfig(
        text_config=LlamaConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
        ),
        vision_config=CLIPVisionConfig(
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=2,
            num_attention_heads=2,
            image_size=16,
            patch_size=8,
        ),
        connector_config=connector_config(name),
        vision_feature_layer=[-2, -1],
        image_token_index=31,
    )
    model = TinyLlavaForConditionalGeneration(config).eval()
    count = 4 if name == "identity" else 3
    ids = torch.tensor([[1] + [31] * count + [2]])
    pixels = torch.randn(1, 3, 16, 16)
    result = model(input_ids=ids, pixel_values=pixels, labels=ids)
    result.loss.backward()
    assert model.model.vision_tower.embeddings.patch_embedding.weight.grad is not None
    for parameter in model.model.multi_modal_projector.parameters():
        assert parameter.grad is not None
    features = model.model.get_image_features(
        pixels, image_sizes=torch.tensor([[16, 16]])
    )
    assert features.pooler_output[0].shape == (count, 16)
    model.save_pretrained(tmp_path)
    restored = TinyLlavaForConditionalGeneration.from_pretrained(tmp_path).eval()
    with torch.no_grad():
        torch.testing.assert_close(
            restored(input_ids=ids, pixel_values=pixels).logits, result.logits
        )


def test_identity_rejects_incompatible_widths():
    with pytest.raises(ValueError, match="matching vision and text"):
        AutoConnectorModel.from_config(
            connector_config("identity"),
            vision_hidden_size=768,
            text_hidden_size=8,
            vision_feature_layer=-1,
        )


def test_connector_selection_is_lazy():
    script = """
import sys
from tinyllava.model.connector import CONNECTOR_CONFIG_MAPPING, AutoConnectorModel
config = CONNECTOR_CONFIG_MAPPING['qformer__tlf_connector']()
assert 'tinyllava.model.connector.qformer.modeling_qformer' not in sys.modules
assert 'transformers.models.blip_2.modeling_blip_2' not in sys.modules
assert 'tinyllava.model.connector.resampler.modeling_resampler' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
