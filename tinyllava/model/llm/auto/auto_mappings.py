"""Project language models; native models remain in the live HF mappings."""

from collections import OrderedDict

LANGUAGE_CONFIG_MAPPING_NAMES = OrderedDict(openelm="OpenELMConfig")
LANGUAGE_MODEL_MAPPING_NAMES = OrderedDict(openelm="OpenELMModel")
LANGUAGE_MODEL_FOR_CAUSAL_LM_MAPPING_NAMES = OrderedDict(openelm="OpenELMForCausalLM")
