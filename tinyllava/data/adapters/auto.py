"""Registry and auto-detection for training dataset adapters."""

from collections.abc import Mapping
from typing import Any

from .base import TrainingDatasetAdapter
from .llava_legacy import LlavaLegacyDatasetAdapter


_ADAPTERS: dict[str, type[TrainingDatasetAdapter]] = {
    LlavaLegacyDatasetAdapter.name: LlavaLegacyDatasetAdapter,
}


def resolve_dataset_adapter(
    name: str,
    first_sample: Mapping[str, Any],
) -> TrainingDatasetAdapter | None:
    """Resolve an explicit adapter or infer one from a representative row."""
    if name == "auto":
        if "conversations" in first_sample:
            name = LlavaLegacyDatasetAdapter.name
        else:
            return None
    try:
        return _ADAPTERS[name]()
    except KeyError as exc:
        raise ValueError(
            f"Unknown training dataset adapter {name!r}. "
            f"Available: {sorted(_ADAPTERS)}"
        ) from exc


def register_dataset_adapter(
    name: str,
    adapter_class: type[TrainingDatasetAdapter],
) -> None:
    _ADAPTERS[name] = adapter_class


__all__ = ["register_dataset_adapter", "resolve_dataset_adapter"]
