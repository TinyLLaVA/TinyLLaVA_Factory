"""HF-style lazy dispatch for project vision models and native HF backbones."""

from __future__ import annotations

import importlib
from collections import OrderedDict
from collections.abc import Iterator

from transformers import AutoModel, PreTrainedConfig, PreTrainedModel
from transformers.models.auto.auto_factory import _BaseAutoModelClass, _LazyAutoMapping

from ..mof.configuration_mof import (
    MofVisionConfig,  # noqa: F401 - HF AutoConfig registration
)


class _LazyAutoVisionTowerMapping(_LazyAutoMapping):
    def __getitem__(self, key: type[PreTrainedConfig]) -> type[PreTrainedModel]:
        if key in self._extra_content:
            return self._extra_content[key]
        # Consult the live HF mapping so public AutoModel registrations work too.
        if key in AutoModel._model_mapping:
            return AutoModel._model_mapping[key]
        return super().__getitem__(key)

    def __contains__(self, key: object) -> bool:
        return key in AutoModel._model_mapping or super().__contains__(key)

    def __iter__(self) -> Iterator[type[PreTrainedConfig]]:
        return iter(self.keys())

    def keys(self) -> list[type[PreTrainedConfig]]:
        return list(dict.fromkeys([*super().keys(), *AutoModel._model_mapping.keys()]))

    def values(self) -> list[type[PreTrainedModel]]:
        return [self[key] for key in self.keys()]

    def items(self) -> list[tuple[type[PreTrainedConfig], type[PreTrainedModel]]]:
        return [(key, self[key]) for key in self.keys()]

    def __len__(self) -> int:
        native = AutoModel._model_mapping
        return len(native) + sum(key not in native for key in super().keys())

    def _load_attr_from_module(
        self, model_type: str, attr: str
    ) -> type[PreTrainedConfig] | type[PreTrainedModel]:
        if model_type not in self._modules:
            self._modules[model_type] = importlib.import_module(
                f".{model_type}",
                "tinyllava.model.vision_tower",
            )
        return getattr(self._modules[model_type], attr)


VISION_TOWER_MODEL_MAPPING = _LazyAutoVisionTowerMapping(
    OrderedDict(mof="MofVisionConfig"),
    OrderedDict(mof="MofVisionModel"),
)


class AutoVisionTowerModel(_BaseAutoModelClass):
    _model_mapping = VISION_TOWER_MODEL_MAPPING


__all__ = ["AutoVisionTowerModel"]
