"""Lazy auto factory for dataset loader adapters."""

from __future__ import annotations

import importlib
from typing import cast

from tinyllava.eval.tasks.auto.auto_mappings import (
    DATASET_LOADER_MAPPING_NAMES,
)
from tinyllava.eval.tasks.loader_base import DatasetLoader


class _LazyDatasetLoaderMapping:
    """Lazy mapping that imports dataset loader modules only when requested."""

    def __init__(self, mapping: dict[str, str]):
        self._mapping = mapping
        self._modules = {}
        self._extra_content = {}

    def __getitem__(self, key: str) -> type[DatasetLoader]:
        if key in self._extra_content:
            return cast(type[DatasetLoader], self._extra_content[key])
        if key not in self._mapping:
            raise KeyError(key)

        if key not in self._modules:
            self._modules[key] = importlib.import_module(
                f".{key}.loader_{key}", "tinyllava.eval.tasks"
            )
        return cast(
            type[DatasetLoader],
            getattr(self._modules[key], self._mapping[key]),
        )

    def keys(self):
        return self._mapping.keys()

    def register(self, key: str, value: type[DatasetLoader]) -> None:
        self._extra_content[key] = value


DATASET_LOADER_MAPPING = _LazyDatasetLoaderMapping(DATASET_LOADER_MAPPING_NAMES)


class AutoDatasetLoader:
    _loader_mapping = DATASET_LOADER_MAPPING

    @classmethod
    def from_name(cls, name: str) -> DatasetLoader:
        try:
            return cls._loader_mapping[name]()
        except KeyError as exc:
            raise ValueError(
                f"Unknown dataset loader adapter {name!r}. "
                f"Available: {sorted(cls._loader_mapping.keys())}"
            ) from exc

    @classmethod
    def register(cls, name: str, adapter_class: type[DatasetLoader]) -> None:
        cls._loader_mapping.register(name, adapter_class)
