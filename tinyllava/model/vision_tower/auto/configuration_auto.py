"""Auto Connector Config mapping."""

from collections import OrderedDict

from transformers.models.auto.configuration_auto import _LazyConfigMapping
from transformers.models.auto.auto_mappings import CONFIG_MAPPING_NAMES

from .auto_mappings import VISION_TOWER_CONFIG_MAPPING_NAMES


VISION_TOWER_CONFIG_MAPPING_NAMES = OrderedDict(
    **VISION_TOWER_CONFIG_MAPPING_NAMES,
    **CONFIG_MAPPING_NAMES,
)

VISION_TOWER_CONFIG_MAPPING = _LazyConfigMapping(VISION_TOWER_CONFIG_MAPPING_NAMES)


__all__ = ["VISION_TOWER_CONFIG_MAPPING"]
