"""Project language models; native model names are merged by the Auto modules."""

from collections import OrderedDict

LANGUAGE_CONFIG_MAPPING_NAMES = OrderedDict(openelm="OpenELMConfig")
LANGUAGE_MODEL_MAPPING_NAMES = OrderedDict(openelm="OpenELMModel")
LANGUAGE_MODEL_FOR_CAUSAL_LM_MAPPING_NAMES = OrderedDict(openelm="OpenELMForCausalLM")
