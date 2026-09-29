"""Lazy project image processors with native Hugging Face fallback."""

from __future__ import annotations

import importlib
from os import PathLike

from transformers import AutoImageProcessor as HFAutoImageProcessor
from transformers.image_processing_base import ImageProcessingMixin
from transformers.image_processing_utils import BaseImageProcessor

from .auto_mappings import CUSTOM_IMAGE_PROCESSOR_MAPPING_NAMES


class AutoImageProcessor(HFAutoImageProcessor):
    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str | PathLike[str], **kwargs
    ) -> BaseImageProcessor:
        """Load the saved image processor, importing a local extension only if selected.

        Project entries use unique class names in `image_processor_type`.
        Native processors, registered HF extensions and remote-code handling
        remain delegated to Hugging Face. This class does not modify HF mappings.
        """
        if CUSTOM_IMAGE_PROCESSOR_MAPPING_NAMES:
            try:
                processor_dict, _ = ImageProcessingMixin.get_image_processor_dict(
                    pretrained_model_name_or_path,
                    **kwargs,
                )
            except OSError:
                # HF also supports sources with preprocessing embedded in model
                # config, such as timm; let its loader resolve these sources.
                return super().from_pretrained(pretrained_model_name_or_path, **kwargs)
            processor_name = processor_dict.get("image_processor_type")
            for module_name, class_name in CUSTOM_IMAGE_PROCESSOR_MAPPING_NAMES.items():
                if processor_name == class_name:
                    module = importlib.import_module(
                        f".{module_name.removesuffix('__tlf_image_processor')}",
                        "tinyllava.data.image_processor",
                    )
                    processor_cls = getattr(module, class_name)
                    return processor_cls.from_pretrained(
                        pretrained_model_name_or_path, **kwargs
                    )
        return super().from_pretrained(pretrained_model_name_or_path, **kwargs)


__all__ = ["AutoImageProcessor"]
