"""Auto Connector Config mapping."""

from typing import cast
from collections import OrderedDict

from transformers.models.auto.configuration_auto import _LazyConfigMapping
from transformers.models.auto.auto_mappings import CONFIG_MAPPING_NAMES

from ..configuration_base import _BaseConnectorConfig
from .auto_mappings import CONNECTOR_CONFIG_MAPPING_NAMES


class _LazyConnectorConfigMapping(_LazyConfigMapping):
    """
    A lazy config mapping that only imports the connector config when requested.
    """

    def __getitem__(self, key: str) -> type[_BaseConnectorConfig]:
        return cast(type[_BaseConnectorConfig], super().__getitem__(key))


CONNECTOR_CONFIG_MAPPING_NAMES = OrderedDict(
    **CONNECTOR_CONFIG_MAPPING_NAMES,
    **CONFIG_MAPPING_NAMES,
)

CONNECTOR_CONFIG_MAPPING = _LazyConnectorConfigMapping(CONNECTOR_CONFIG_MAPPING_NAMES)


__all__ = ["CONNECTOR_CONFIG_MAPPING"]
