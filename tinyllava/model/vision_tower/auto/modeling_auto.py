"""Auto Vision Tower Model class."""

import importlib
from collections import OrderedDict

from transformers.models.auto.auto_factory import _BaseAutoModelClass, _LazyAutoMapping
from transformers.models.auto.modeling_auto import MODEL_MAPPING_NAMES

from .auto_mappings import VISION_TOWER_MODEL_MAPPING_NAMES
from .configuration_auto import VISION_TOWER_CONFIG_MAPPING_NAMES


class _LazyAutoVisionTowerModelMapping(_LazyAutoMapping):
    """
    A mapping config to object (vision tower for instance) that will load keys and values when it is accessed.

    Args:
        - config_mapping: The map model type to config class
        - model_mapping: The map model type to vision tower class
    """

    def _load_attr_from_module(self, model_type, attr):
        if model_type.endswith("__tlf_vision_tower"):
            module_name = model_type.replace("__tlf_vision_tower", "")
            if module_name not in self._modules:
                self._modules[module_name] = importlib.import_module(f".{module_name}", "tinyllava.model.vision_tower")
            return getattr(self._modules[module_name], attr)

        return super()._load_attr_from_module(model_type, attr)


VISION_TOWER_MODEL_MAPPING_NAMES = OrderedDict(
    **VISION_TOWER_MODEL_MAPPING_NAMES,
    **MODEL_MAPPING_NAMES,
)

VISION_TOWER_MODEL_MAPPING = _LazyAutoVisionTowerModelMapping(VISION_TOWER_CONFIG_MAPPING_NAMES, VISION_TOWER_MODEL_MAPPING_NAMES)


class AutoVisionTowerModel(_BaseAutoModelClass):
    _model_mapping = VISION_TOWER_MODEL_MAPPING


__all__ = ["AutoVisionTowerModel"]
