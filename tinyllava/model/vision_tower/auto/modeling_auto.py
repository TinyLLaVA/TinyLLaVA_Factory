"""HF Auto dispatch with project-specific lazy module resolution."""

import importlib
from collections import OrderedDict

from transformers.models.auto.auto_factory import _BaseAutoModelClass, _LazyAutoMapping
from transformers.models.auto.modeling_auto import MODEL_MAPPING_NAMES

from ..mof.configuration_mof import (
    MofVisionConfig,  # noqa: F401 - register saved config with HF AutoConfig
)
from . import auto_mappings
from .auto_mappings import VISION_TOWER_MODEL_MAPPING_NAMES
from .configuration_auto import VISION_TOWER_CONFIG_MAPPING_NAMES


class _LazyAutoVisionTowerModelMapping(_LazyAutoMapping):
    def _load_attr_from_module(self, model_type: str, attr: str) -> type:
        if model_type in auto_mappings.VISION_TOWER_CONFIG_MAPPING_NAMES:
            module_name = model_type.removesuffix("__tlf_vision_tower")
            if model_type not in self._modules:
                self._modules[model_type] = importlib.import_module(
                    f".{module_name}", "tinyllava.model.vision_tower"
                )
            return getattr(self._modules[model_type], attr)
        return super()._load_attr_from_module(model_type, attr)


VISION_TOWER_MODEL_MAPPING_NAMES = OrderedDict(
    {**MODEL_MAPPING_NAMES, **VISION_TOWER_MODEL_MAPPING_NAMES}
)
VISION_TOWER_MODEL_MAPPING = _LazyAutoVisionTowerModelMapping(
    VISION_TOWER_CONFIG_MAPPING_NAMES, VISION_TOWER_MODEL_MAPPING_NAMES
)


class AutoVisionTowerModel(_BaseAutoModelClass):
    _model_mapping = VISION_TOWER_MODEL_MAPPING


__all__ = ["AutoVisionTowerModel"]
