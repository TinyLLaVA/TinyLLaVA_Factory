"""Auto Connector Config mapping."""

import importlib
from typing import cast

from transformers.models.auto.configuration_auto import _LazyConfigMapping

from ..configuration_base import BaseConnectorConfig
from .auto_mappings import CONNECTOR_CONFIG_MAPPING_NAMES


class _LazyConnectorConfigMapping(_LazyConfigMapping):
    """
    A lazy config mapping that only imports the connector config when requested.
    """

    def __getitem__(self, key: str) -> type[BaseConnectorConfig]:
        if key in self._extra_content:
            return cast(type[BaseConnectorConfig], self._extra_content[key])
        if key not in self._mapping or not key.endswith("__tlf_connector"):
            raise KeyError(key)

        module_name = key.removesuffix("__tlf_connector")
        if module_name not in self._modules:
            self._modules[module_name] = importlib.import_module(
                f".{module_name}", "tinyllava.model.connector"
            )
        return cast(
            type[BaseConnectorConfig],
            getattr(self._modules[module_name], self._mapping[key]),
        )


CONNECTOR_CONFIG_MAPPING = _LazyConnectorConfigMapping(CONNECTOR_CONFIG_MAPPING_NAMES)


__all__ = ["CONNECTOR_CONFIG_MAPPING"]
