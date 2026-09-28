"""Lazy model-config dispatch for project multimodal processors."""

from __future__ import annotations

import importlib
from os import PathLike

from transformers import AutoProcessor as HFAutoProcessor
from transformers.models.auto.auto_factory import _LazyAutoMapping
from transformers.models.auto.processing_auto import (
    PROCESSOR_MAPPING as HF_PROCESSOR_MAPPING,
)
from transformers.processing_utils import ProcessorMixin

from tinyllava.model.configuration_tinyllava import TinyLlavaConfig

from .auto_mappings import PROCESSOR_CONFIG_MAPPING_NAMES, PROCESSOR_MAPPING_NAMES


class _LazyProcessorMapping(_LazyAutoMapping):
    def _load_attr_from_module(self, model_type: str, attr: str) -> type:
        if attr == PROCESSOR_CONFIG_MAPPING_NAMES[model_type]:
            module = importlib.import_module(
                f"tinyllava.model.configuration_{model_type}"
            )
        else:
            module = importlib.import_module(f"tinyllava.data.processor.{model_type}")
        return getattr(module, attr)


PROCESSOR_MAPPING = _LazyProcessorMapping(
    PROCESSOR_CONFIG_MAPPING_NAMES,
    PROCESSOR_MAPPING_NAMES,
)


class AutoProcessor(HFAutoProcessor):
    @classmethod
    def from_config(cls, config: TinyLlavaConfig, **kwargs) -> ProcessorMixin:
        """Select the processor by composite model type, independent of connector."""
        if type(config) in PROCESSOR_MAPPING:
            return PROCESSOR_MAPPING[type(config)].from_config(config, **kwargs)
        return HF_PROCESSOR_MAPPING[type(config)](**kwargs)

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str | PathLike[str], **kwargs
    ) -> ProcessorMixin:
        """Load a saved project processor lazily, delegating native classes to HF."""
        processor_dict, _ = ProcessorMixin.get_processor_dict(
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
