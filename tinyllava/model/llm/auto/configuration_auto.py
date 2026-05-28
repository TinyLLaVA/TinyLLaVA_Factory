"""Auto Language Config mapping."""

from collections import OrderedDict

from transformers.models.auto.configuration_auto import _LazyConfigMapping
from transformers.models.auto.auto_mappings import CONFIG_MAPPING_NAMES

from .auto_mappings import LANGUAGE_CONFIG_MAPPING_NAMES


LANGUAGE_CONFIG_MAPPING_NAMES = OrderedDict(
    **LANGUAGE_CONFIG_MAPPING_NAMES,
    **CONFIG_MAPPING_NAMES,
)

LANGUAGE_CONFIG_MAPPING = _LazyConfigMapping(LANGUAGE_CONFIG_MAPPING_NAMES)


__all__ = ["LANGUAGE_CONFIG_MAPPING"]
