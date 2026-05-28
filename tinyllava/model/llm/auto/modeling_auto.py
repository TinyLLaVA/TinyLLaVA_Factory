"""Auto Language Model class."""

import importlib
from collections import OrderedDict

from transformers.models.auto.auto_factory import _BaseAutoModelClass, _LazyAutoMapping
from transformers.models.auto.modeling_auto import MODEL_MAPPING_NAMES

from .auto_mappings import LANGUAGE_MODEL_MAPPING_NAMES
from .configuration_auto import LANGUAGE_CONFIG_MAPPING_NAMES


class _LazyAutoLanguageModelMapping(_LazyAutoMapping):
    """
    A mapping config to object (language model for instance) that will load keys and values when it is accessed.

    Args:
        - config_mapping: The map model type to config class
        - model_mapping: The map model type to language model class
    """

    def _load_attr_from_module(self, model_type, attr):
        if model_type.endswith("__tlf_language_model"):
            module_name = model_type.replace("__tlf_language_model", "")
            if module_name not in self._modules:
                self._modules[module_name] = importlib.import_module(f".{module_name}", "tinyllava.model.llm")
            return getattr(self._modules[module_name], attr)

        return super()._load_attr_from_module(model_type, attr)


LANGUAGE_MODEL_MAPPING_NAMES = OrderedDict(
    **LANGUAGE_MODEL_MAPPING_NAMES,
    **MODEL_MAPPING_NAMES,
)

LANGUAGE_MODEL_MAPPING = _LazyAutoLanguageModelMapping(LANGUAGE_CONFIG_MAPPING_NAMES, LANGUAGE_MODEL_MAPPING_NAMES)


class AutoLanguageModel(_BaseAutoModelClass):
    _model_mapping = LANGUAGE_MODEL_MAPPING


__all__ = ["AutoLanguageModel"]
