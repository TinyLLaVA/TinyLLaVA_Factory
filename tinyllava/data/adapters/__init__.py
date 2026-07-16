"""Source-specific adapters for canonical training samples."""

from .auto import register_dataset_adapter, resolve_dataset_adapter
from .base import TrainingDatasetAdapter
from .llava_legacy import LlavaLegacyDatasetAdapter

__all__ = [
    "LlavaLegacyDatasetAdapter",
    "TrainingDatasetAdapter",
    "register_dataset_adapter",
    "resolve_dataset_adapter",
]
