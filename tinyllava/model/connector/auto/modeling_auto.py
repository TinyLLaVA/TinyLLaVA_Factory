"""Auto Connector Model class."""

import importlib

from transformers.models.auto.auto_factory import _BaseAutoModelClass, _LazyAutoMapping

from .configuration_auto import CONNECTOR_CONFIG_MAPPING_NAMES
from .auto_mappings import CONNECTOR_MODEL_MAPPING_NAMES


class _LazyAutoConnectorModelMapping(_LazyAutoMapping):
    """
    A mapping config to object (connector model for instance) that will load keys and values when it is accessed.

    Args:
        - config_mapping: The map model type to config class
        - model_mapping: The map model type to connector model class
    """

    def _load_attr_from_module(self, model_type, attr):
        if not model_type.endswith("__tlf_connector"):
            raise KeyError(model_type)

        module_name = model_type.removesuffix("__tlf_connector")
        if module_name not in self._modules:
            self._modules[module_name] = importlib.import_module(
                f".{module_name}", "tinyllava.model.connector"
            )
        return getattr(self._modules[module_name], attr)


CONNECTOR_MODEL_MAPPING = _LazyAutoConnectorModelMapping(CONNECTOR_CONFIG_MAPPING_NAMES, CONNECTOR_MODEL_MAPPING_NAMES)


class AutoConnectorModel(_BaseAutoModelClass):
    _model_mapping = CONNECTOR_MODEL_MAPPING


__all__ = ["AutoConnectorModel"]
