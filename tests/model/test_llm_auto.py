import subprocess
import sys

import pytest
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    LlamaConfig,
    LlamaForCausalLM,
    LlamaModel,
)

from tinyllava.model import TinyLlavaConfig
from tinyllava.model.llm import (
    LANGUAGE_CONFIG_MAPPING,
    AutoLanguageModel,
    AutoLanguageModelForCausalLM,
)


def test_language_auto_observes_public_hf_registrations():
    class RegisteredConfig(LlamaConfig):
        model_type = "tinyllava_test_language"

    class RegisteredModel(LlamaModel):
        config_class = RegisteredConfig

    class RegisteredLM(LlamaForCausalLM):
        config_class = RegisteredConfig

    AutoConfig.register(RegisteredConfig.model_type, RegisteredConfig)
    AutoModel.register(RegisteredConfig, RegisteredModel)
    AutoModelForCausalLM.register(RegisteredConfig, RegisteredLM)
    config = LANGUAGE_CONFIG_MAPPING[RegisteredConfig.model_type](
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
    )
    assert isinstance(AutoLanguageModel.from_config(config), RegisteredModel)
    assert isinstance(AutoLanguageModelForCausalLM.from_config(config), RegisteredLM)


def test_text_config_dictionary_is_not_modified():
    config = {"hidden_size": 8, "num_attention_heads": 2, "num_key_value_heads": 2}
    TinyLlavaConfig(text_config=config)
    assert "model_type" not in config


@pytest.mark.parametrize("selection", ["llama", "openelm"])
def test_language_selection_imports_only_selected_implementation(selection):
    script = """
import sys
from tinyllava.model.llm import LANGUAGE_CONFIG_MAPPING, AutoLanguageModel
assert 'tinyllava.model.llm.openelm.modeling_openelm' not in sys.modules
assert 'transformers.models.llama.modeling_llama' not in sys.modules
name = sys.argv[1]
kwargs = dict(vocab_size=32, model_dim=16, head_dim=4, num_transformer_layers=1, ffn_dim_divisor=8) if name == 'openelm' else dict(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2)
c = LANGUAGE_CONFIG_MAPPING[name](**kwargs)
assert 'tinyllava.model.llm.openelm.modeling_openelm' not in sys.modules
AutoLanguageModel.from_config(c)
assert ('tinyllava.model.llm.openelm.modeling_openelm' in sys.modules) == (name == 'openelm')
assert ('transformers.models.llama.modeling_llama' in sys.modules) == (name == 'llama')
"""
    result = subprocess.run(
        [sys.executable, "-c", script, selection], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("model_type", ["llama", "gemma", "phi", "qwen2", "stablelm"])
def test_native_language_models_use_hf_classes(model_type):
    import torch

    config = AutoConfig.for_model(
        model_type,
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
    )
    backbone = AutoLanguageModel.from_config(config).eval()
    causal_lm = AutoLanguageModelForCausalLM.from_config(config).eval()
    assert type(backbone) is AutoModel._model_mapping[type(config)]
    assert type(causal_lm) is AutoModelForCausalLM._model_mapping[type(config)]
    with torch.no_grad():
        assert backbone(torch.tensor([[1, 2]])).last_hidden_state.shape == (1, 2, 16)
        assert causal_lm(torch.tensor([[1, 2]])).logits.shape == (1, 2, 32)
