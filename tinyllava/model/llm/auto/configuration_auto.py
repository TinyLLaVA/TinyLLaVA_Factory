"""Lazy project configurations with live Hugging Face registration fallback."""

from __future__ import annotations

import importlib
from collections.abc import Iterator

from transformers import PreTrainedConfig
from transformers.models.auto.configuration_auto import (
    CONFIG_MAPPING,
    _LazyConfigMapping,
)

from .auto_mappings import LANGUAGE_CONFIG_MAPPING_NAMES


class _LazyLanguageConfigMapping(_LazyConfigMapping):
    def __getitem__(self, key: str) -> type[PreTrainedConfig]:
        if key in self._extra_content:
            return self._extra_content[key]
        if key in CONFIG_MAPPING:
            return CONFIG_MAPPING[key]
        if key not in self._mapping:
            raise KeyError(key)
        if key not in self._modules:
            self._modules[key] = importlib.import_module(f"tinyllava.model.llm.{key}")
        return getattr(self._modules[key], self._mapping[key])

    def __contains__(self, key: object) -> bool:
        return key in CONFIG_MAPPING or super().__contains__(key)

    def keys(self) -> list[str]:
        return list(dict.fromkeys([*super().keys(), *CONFIG_MAPPING.keys()]))

    def __iter__(self) -> Iterator[str]:
        return iter(self.keys())

    def __len__(self) -> int:
        return len(self.keys())

    def values(self) -> list[type[PreTrainedConfig]]:
        return [self[key] for key in self.keys()]

    def items(self) -> list[tuple[str, type[PreTrainedConfig]]]:
        return [(key, self[key]) for key in self.keys()]


LANGUAGE_CONFIG_MAPPING = _LazyLanguageConfigMapping(LANGUAGE_CONFIG_MAPPING_NAMES)

__all__ = ["LANGUAGE_CONFIG_MAPPING"]
