from types import SimpleNamespace

import pytest
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

from tinyllava.hparams.arguments import TuningArguments
from tinyllava.model.configuration_tinyllava import TinyLlavaConfig
from tinyllava.model.modeling_tinyllava import TinyLlavaForConditionalGeneration
from tinyllava.train.strategy.peft_strategy import LoraInt8TrainingStrategy
from tinyllava.utils.precision import cast_model_to_training_dtype


@pytest.mark.parametrize("quantized", [False, True])
def test_from_pretrained_components_loads_causal_head_and_vision_only(
    tmp_path, quantized
):
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
        language_model_loading_kwargs=(
            LoraInt8TrainingStrategy(
                SimpleNamespace(bf16=False, fp16=False), TuningArguments()
            ).language_model_loading_kwargs()
            if quantized
            else {}
        ),
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
    if quantized:
        from peft.tuners.lora.bnb import Linear8bitLt

        assert model.is_loaded_in_8bit
        model = LoraInt8TrainingStrategy(
            None, TuningArguments(tune_type_llm="lora", peft={"r": 2})
        )(model)
        cast_model_to_training_dtype(model, SimpleNamespace(bf16=False, fp16=False))
        assert isinstance(
            model.get_base_model().model.language_model.layers[0].self_attn.q_proj,
            Linear8bitLt,
        )
        ids = torch.tensor([[1, 2, 3]])
        model(input_ids=ids, labels=ids).loss.backward()
        assert (
            model.get_base_model()
            .model.language_model.layers[0]
            .self_attn.q_proj.lora_B.default.weight.grad
            is not None
        )
