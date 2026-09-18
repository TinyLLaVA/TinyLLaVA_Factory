"""Registry and auto-detection for training dataset adapters."""

from collections.abc import Mapping
from typing import Any

from datasets import Features, Value

from .base import TrainingDatasetAdapter
from .llava_legacy import LlavaLegacyDatasetAdapter


_ADAPTERS: dict[str, type[TrainingDatasetAdapter]] = {
    LlavaLegacyDatasetAdapter.name: LlavaLegacyDatasetAdapter,
}


def resolve_dataset_adapter(
    name: str,
    first_sample: Mapping[str, Any],
    *,
    features: Features | None = None,
) -> TrainingDatasetAdapter | None:
    """Resolve an explicit adapter or infer one from a representative row."""
    if name == "auto":
        # The legacy adapter projects image paths into a string column. Native
        # messages and decoded/embedded media must retain their original schema.
        if "messages" in first_sample or "images" in first_sample:
            return None
        if not isinstance(first_sample.get("image"), (str, type(None))):
            return None
        if (
            features is not None
            and "image" in features
            and features["image"]
            not in (
                Value("string"),
                Value("null"),
            )
        ):
            return None
        if "conversations" in first_sample:
            if any(
                isinstance(message, Mapping)
                and isinstance(message.get("content", message.get("value")), list)
                for message in first_sample["conversations"]
            ):
                return None
            name = LlavaLegacyDatasetAdapter.name
        else:
            return None
    try:
        return _ADAPTERS[name]()
    except KeyError as exc:
        raise ValueError(
            f"Unknown training dataset adapter {name!r}. Available: {sorted(_ADAPTERS)}"
        ) from exc


def register_dataset_adapter(
    name: str,
    adapter_class: type[TrainingDatasetAdapter],
) -> None:
    _ADAPTERS[name] = adapter_class


__all__ = ["register_dataset_adapter", "resolve_dataset_adapter"]
