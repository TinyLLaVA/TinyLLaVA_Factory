"""LLaVA-derived processor registered for TinyLLaVA model configs."""

from transformers import AutoProcessor

from tinyllava.model.configuration_tinyllava import TinyLlavaConfig

from ..processing_base import BaseProcessor


class TinyLlavaProcessor(BaseProcessor):
    """Bind shared LLaVA preprocessing to the TinyLLaVA Auto mapping."""


AutoProcessor.register(TinyLlavaConfig, TinyLlavaProcessor, exist_ok=True)

__all__ = ["TinyLlavaProcessor"]
