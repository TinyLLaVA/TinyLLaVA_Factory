import torch
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    SiglipConfig,
    SiglipModel,
    SiglipTextConfig,
    SiglipVisionConfig,
    SiglipVisionModel,
)

from tinyllava.model.configuration_tinyllava import TinyLlavaConfig
from tinyllava.model.modeling_tinyllava import TinyLlavaForConditionalGeneration


def test_from_pretrained_components_loads_causal_head_and_vision_only(tmp_path):
    language_path = tmp_path / "language"
    vision_path = tmp_path / "vision"

    text_config = LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        tie_word_embeddings=False,
    )
    causal_lm = LlamaForCausalLM(text_config)
    causal_lm.save_pretrained(language_path)

    siglip_vision_config = SiglipVisionConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        image_size=16,
        patch_size=8,
    )
    siglip_config = SiglipConfig(
        text_config=SiglipTextConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
        ).to_dict(),
        vision_config=siglip_vision_config.to_dict(),
    )
    siglip = SiglipModel(siglip_config)
    siglip.save_pretrained(vision_path)

    config = TinyLlavaConfig(
        text_config=text_config,
        vision_config=siglip_vision_config,
        vision_feature_select_strategy="full",
    )
    model = TinyLlavaForConditionalGeneration.from_pretrained_components(
        config,
        language_model_name_or_path=str(language_path),
        vision_model_name_or_path=str(vision_path),
    )

    assert isinstance(model.model.vision_tower, SiglipVisionModel)
    assert not hasattr(model.model.vision_tower, "text_model")
    assert torch.equal(model.lm_head.weight, causal_lm.lm_head.weight)
    assert torch.equal(
        model.model.language_model.embed_tokens.weight,
        causal_lm.model.embed_tokens.weight,
    )
    assert torch.equal(
        model.model.vision_tower.embeddings.patch_embedding.weight,
        siglip.vision_model.embeddings.patch_embedding.weight,
    )
