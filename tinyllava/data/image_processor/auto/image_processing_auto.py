"""Auto Custom Processor class."""

import importlib
from types import MethodType

from transformers.models.auto.image_processing_auto import (
    IMAGE_PROCESSOR_MAPPING_NAMES as TRANSFORMERS_PROCESSOR_MAPPING_NAMES,
    AutoImageProcessor,
)
from transformers.models.auto import image_processing_auto as transformers_image_processing_auto

from .auto_mappings import CUSTOM_PROCESSOR_MAPPING_NAMES


def _load_custom_processor_attr_from_module(self, model_type, attr):
    if model_type.endswith("__tlf_image_processor"):
        module_name = model_type.replace("__tlf_image_processor", "")
        if module_name not in self._modules:
            self._modules[module_name] = importlib.import_module(f".{module_name}", "tinyllava.data.image_processor")
        return getattr(self._modules[module_name], attr)

    return self._transformers_original_load_attr_from_module(model_type, attr)


def _register_custom_processor_mappings() -> None:
    """Inject TinyLLaVA processor names into Transformers' global auto mapping."""
    TRANSFORMERS_PROCESSOR_MAPPING_NAMES.update(CUSTOM_PROCESSOR_MAPPING_NAMES)

    image_processor_mapping = transformers_image_processing_auto.IMAGE_PROCESSOR_MAPPING

    # kept for reliability, but not strictly necessary since the mapping is updated in place
    image_processor_mapping._model_mapping = TRANSFORMERS_PROCESSOR_MAPPING_NAMES

    if not hasattr(image_processor_mapping, "_transformers_original_load_attr_from_module"):
        image_processor_mapping._transformers_original_load_attr_from_module = image_processor_mapping._load_attr_from_module

    image_processor_mapping._load_attr_from_module = MethodType(
        _load_custom_processor_attr_from_module,
        image_processor_mapping,
    )


_register_custom_processor_mappings()


__all__ = ["AutoImageProcessor"]
