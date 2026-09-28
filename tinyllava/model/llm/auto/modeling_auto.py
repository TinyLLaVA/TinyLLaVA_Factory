"""Lazy language backbone and causal-LM dispatch with live HF fallback."""

from __future__ import annotations

import importlib
from collections import OrderedDict
from collections.abc import Iterator

from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    PreTrainedConfig,
    PreTrainedModel,
)
from transformers.models.auto.auto_factory import _BaseAutoModelClass, _LazyAutoMapping

from ..openelm.configuration_openelm import (
    OpenELMConfig,  # noqa: F401 - HF AutoConfig registration
)
from .auto_mappings import (
    LANGUAGE_CONFIG_MAPPING_NAMES,
    LANGUAGE_MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    LANGUAGE_MODEL_MAPPING_NAMES,
)


class _LazyAutoLanguageModelMapping(_LazyAutoMapping):
    def __init__(
        self,
        model_mapping: OrderedDict[str, str],
        native_auto: type[_BaseAutoModelClass],
    ):
        self.native_auto = native_auto
        super().__init__(LANGUAGE_CONFIG_MAPPING_NAMES, model_mapping)

    def __getitem__(self, key: type[PreTrainedConfig]) -> type[PreTrainedModel]:
        if key in self._extra_content:
            return self._extra_content[key]
        # Consult the live HF mapping so public AutoModel registrations work too.
        if key in self.native_auto._model_mapping:
            return self.native_auto._model_mapping[key]
        return super().__getitem__(key)

    def __contains__(self, key: object) -> bool:
        return key in self.native_auto._model_mapping or super().__contains__(key)

    def __iter__(self) -> Iterator[type[PreTrainedConfig]]:
        return iter(self.keys())

    def keys(self) -> list[type[PreTrainedConfig]]:
        return list(
            dict.fromkeys([*super().keys(), *self.native_auto._model_mapping.keys()])
        )

    def values(self) -> list[type[PreTrainedModel]]:
        return [self[key] for key in self.keys()]

    def items(self) -> list[tuple[type[PreTrainedConfig], type[PreTrainedModel]]]:
        return [(key, self[key]) for key in self.keys()]

    def __len__(self) -> int:
        native = self.native_auto._model_mapping
        return len(native) + sum(key not in native for key in super().keys())

    def _load_attr_from_module(
        self, model_type: str, attr: str
    ) -> type[PreTrainedConfig] | type[PreTrainedModel]:
        if model_type not in self._modules:
            self._modules[model_type] = importlib.import_module(
                f".{model_type}",
                "tinyllava.model.llm",
            )
        return getattr(self._modules[model_type], attr)


LANGUAGE_MODEL_MAPPING = _LazyAutoLanguageModelMapping(
    LANGUAGE_MODEL_MAPPING_NAMES, AutoModel
)
LANGUAGE_MODEL_FOR_CAUSAL_LM_MAPPING = _LazyAutoLanguageModelMapping(
    LANGUAGE_MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    AutoModelForCausalLM,
)


class AutoLanguageModel(_BaseAutoModelClass):
    _model_mapping = LANGUAGE_MODEL_MAPPING


class AutoLanguageModelForCausalLM(_BaseAutoModelClass):
    _model_mapping = LANGUAGE_MODEL_FOR_CAUSAL_LM_MAPPING


__all__ = ["AutoLanguageModel", "AutoLanguageModelForCausalLM"]
