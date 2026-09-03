import torch
from transformers import CLIPVisionConfig, LlamaConfig

from tinyllava.model.configuration_tinyllava import TinyLlavaConfig
from tinyllava.model.modeling_tinyllava import TinyLlavaForConditionalGeneration


def _tiny_model() -> TinyLlavaForConditionalGeneration:
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
        connector_config={"model_type": "mlp__tlf_connector", "depth": 1},
        image_token_index=63,
        image_seq_length=4,
        vision_feature_select_strategy="default",
    )
    return TinyLlavaForConditionalGeneration(config).eval()


def test_forward_supports_text_only_inputs():
    model = _tiny_model()

    with torch.no_grad():
        output = model(input_ids=torch.tensor([[1, 2, 3]]))

    assert output.logits.shape == (1, 3, 64)
    assert output.image_hidden_states is None


def test_forward_injects_pixel_values_at_image_placeholders():
    model = _tiny_model()
    input_ids = torch.tensor([[1, 63, 63, 63, 63, 2]])
    pixel_values = torch.zeros((1, 3, 16, 16))

    with torch.no_grad():
        output = model(input_ids=input_ids, pixel_values=pixel_values)

    assert output.logits.shape == (1, 6, 64)
    assert output.image_hidden_states.shape == (4, 16)
