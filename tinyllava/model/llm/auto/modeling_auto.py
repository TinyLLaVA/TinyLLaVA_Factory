"""HF Auto dispatch with project-specific lazy module resolution."""

import importlib
from collections import OrderedDict

from transformers.models.auto.auto_factory import _BaseAutoModelClass, _LazyAutoMapping
from transformers.models.auto.modeling_auto import (
    MODEL_MAPPING_NAMES,
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
)

from ..openelm.configuration_openelm import (
    OpenELMConfig,  # noqa: F401 - register saved config with HF AutoConfig
)
from . import auto_mappings
from .auto_mappings import (
    LANGUAGE_MODEL_MAPPING_NAMES,
    LANGUAGE_MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
)
from .configuration_auto import LANGUAGE_CONFIG_MAPPING_NAMES


class _LazyAutoLanguageModelMapping(_LazyAutoMapping):
    def _load_attr_from_module(self, model_type: str, attr: str) -> type:
        if model_type in auto_mappings.LANGUAGE_CONFIG_MAPPING_NAMES:
            module_name = model_type.removesuffix("__tlf_language_model")
            if model_type not in self._modules:
                self._modules[model_type] = importlib.import_module(
                    f".{module_name}", "tinyllava.model.llm"
                )
            return getattr(self._modules[model_type], attr)
        return super()._load_attr_from_module(model_type, attr)


LANGUAGE_MODEL_MAPPING_NAMES = OrderedDict(
    {**MODEL_MAPPING_NAMES, **LANGUAGE_MODEL_MAPPING_NAMES}
)
LANGUAGE_MODEL_MAPPING = _LazyAutoLanguageModelMapping(
    LANGUAGE_CONFIG_MAPPING_NAMES, LANGUAGE_MODEL_MAPPING_NAMES
)


class AutoLanguageModel(_BaseAutoModelClass):
    _model_mapping = LANGUAGE_MODEL_MAPPING


LANGUAGE_MODEL_FOR_CAUSAL_LM_MAPPING_NAMES = OrderedDict(
    {**MODEL_FOR_CAUSAL_LM_MAPPING_NAMES, **LANGUAGE_MODEL_FOR_CAUSAL_LM_MAPPING_NAMES}
)
LANGUAGE_MODEL_FOR_CAUSAL_LM_MAPPING = _LazyAutoLanguageModelMapping(
    LANGUAGE_CONFIG_MAPPING_NAMES, LANGUAGE_MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
)


class AutoLanguageModelForCausalLM(_BaseAutoModelClass):
    _model_mapping = LANGUAGE_MODEL_FOR_CAUSAL_LM_MAPPING


__all__ = ["AutoLanguageModel", "AutoLanguageModelForCausalLM"]
