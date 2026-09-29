"""Lazy local configurations alongside native HF configuration names."""

import importlib
from collections import OrderedDict

from transformers.models.auto.configuration_auto import (
    CONFIG_MAPPING_NAMES,
    _LazyConfigMapping,
)

from . import auto_mappings
from .auto_mappings import VISION_TOWER_CONFIG_MAPPING_NAMES


class _LazyVisionTowerConfigMapping(_LazyConfigMapping):
    def __getitem__(self, key: str) -> type:
        if key in self._extra_content:
            return self._extra_content[key]
        if key in auto_mappings.VISION_TOWER_CONFIG_MAPPING_NAMES:
            module_name = key.removesuffix("__tlf_vision_tower")
            if key not in self._modules:
                self._modules[key] = importlib.import_module(
                    f".{module_name}", "tinyllava.model.vision_tower"
                )
            return getattr(self._modules[key], self._mapping[key])
        return super().__getitem__(key)


VISION_TOWER_CONFIG_MAPPING_NAMES = OrderedDict(
    {**CONFIG_MAPPING_NAMES, **VISION_TOWER_CONFIG_MAPPING_NAMES}
)
VISION_TOWER_CONFIG_MAPPING = _LazyVisionTowerConfigMapping(
    VISION_TOWER_CONFIG_MAPPING_NAMES
)

__all__ = ["VISION_TOWER_CONFIG_MAPPING"]
