"""Lazy model-config dispatch for project multimodal processors."""

from __future__ import annotations

import importlib
from os import PathLike

from transformers import AutoProcessor as HFAutoProcessor, PreTrainedConfig
from transformers.models.auto.auto_factory import _LazyAutoMapping
from transformers.models.auto.processing_auto import (
    PROCESSOR_MAPPING as HF_PROCESSOR_MAPPING,
)
from transformers.processing_utils import ProcessorMixin

from .auto_mappings import (
    PROCESSOR_CONFIG_MAPPING_NAMES,
    PROCESSOR_MAPPING_NAMES,
    PROCESSOR_CONFIG_MODULE_NAMES,
    PROCESSOR_MODULE_NAMES,
)


class _LazyProcessorMapping(_LazyAutoMapping):
    def _load_attr_from_module(self, model_type: str, attr: str) -> type:
        is_config = attr == PROCESSOR_CONFIG_MAPPING_NAMES[model_type]
        modules = PROCESSOR_CONFIG_MODULE_NAMES if is_config else PROCESSOR_MODULE_NAMES
        module_path = modules[model_type]
        cache_key = (model_type, module_path)
        if cache_key not in self._modules:
            self._modules[cache_key] = importlib.import_module(module_path)
        return getattr(self._modules[cache_key], attr)


PROCESSOR_MAPPING = _LazyProcessorMapping(
    PROCESSOR_CONFIG_MAPPING_NAMES,
    PROCESSOR_MAPPING_NAMES,
)


class AutoProcessor(HFAutoProcessor):
    @classmethod
    def from_config(cls, config: PreTrainedConfig, **kwargs) -> ProcessorMixin:
        """Select the processor by composite model type, independent of connector."""
        if type(config) in PROCESSOR_MAPPING:
            return PROCESSOR_MAPPING[type(config)](**kwargs)
        return HF_PROCESSOR_MAPPING[type(config)](**kwargs)

    @classmethod
    def register(
        cls,
        config_class: type[PreTrainedConfig],
        processor_class: type[ProcessorMixin],
        exist_ok: bool = False,
    ) -> None:
        """Register a processor for this project Auto entry point."""
        PROCESSOR_MAPPING.register(config_class, processor_class, exist_ok=exist_ok)

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str | PathLike[str], **kwargs
    ) -> ProcessorMixin:
        """Load a saved project processor lazily, delegating native classes to HF."""
        processor_dict, _ = ProcessorMixin.get_processor_dict(
            pretrained_model_name_or_path, **kwargs
        )
        processor_name = processor_dict.get("processor_class")
        for processor_cls in PROCESSOR_MAPPING._extra_content.values():
            if processor_name == processor_cls.__name__:
                return processor_cls.from_pretrained(
                    pretrained_model_name_or_path, **kwargs
                )
        for model_type, class_name in PROCESSOR_MAPPING_NAMES.items():
            if processor_dict.get("processor_class") == class_name:
                processor_cls = PROCESSOR_MAPPING._load_attr_from_module(
                    model_type, class_name
                )
                return processor_cls.from_pretrained(
                    pretrained_model_name_or_path, **kwargs
                )
        return super().from_pretrained(pretrained_model_name_or_path, **kwargs)


__all__ = ["AutoProcessor"]
