"""Lazy training strategy registry."""

import importlib
from collections.abc import Mapping
from typing import Any

from .auto_mappings import TRAINING_STRATEGY_MAPPING_NAMES


def _strategy_to_module_name(strategy_name: str) -> str:
    if strategy_name == "common":
        return "base"
    if strategy_name in {"lora", "lora_int8"}:
        return "peft_strategy"
    return f"{strategy_name}_strategy"


class _LazyTrainingStrategyMapping(Mapping):
    def __init__(self, mapping):
        self._mapping = {key.lower(): value for key, value in mapping.items()}
        self._modules = {}
        self._extra_content = {}

    def __getitem__(self, key: str):
        strategy_name = key.lower()
        if strategy_name in self._extra_content:
            return self._extra_content[strategy_name]
        if strategy_name not in self._mapping:
            raise KeyError(key)

        module_name = _strategy_to_module_name(strategy_name)
        if module_name not in self._modules:
            self._modules[module_name] = importlib.import_module(
                f".{module_name}", "tinyllava.train.strategy"
            )
        return getattr(self._modules[module_name], self._mapping[strategy_name])

    def __iter__(self):
        yield from self._mapping
        yield from self._extra_content

    def __len__(self):
        return len(set(self._mapping) | set(self._extra_content))

    def register(self, key: str, value: Any) -> None:
        self._extra_content[key.lower()] = value


TRAINING_STRATEGY_MAPPING = _LazyTrainingStrategyMapping(
    TRAINING_STRATEGY_MAPPING_NAMES
)


def get_training_strategy(training_strategy: str):
    try:
        return TRAINING_STRATEGY_MAPPING[training_strategy]
    except KeyError as exc:
        available = sorted(TRAINING_STRATEGY_MAPPING)
        raise ValueError(
            f"{training_strategy} is not registered. Available strategies: {available}"
        ) from exc


def register_training_strategy(name):
    def register_training_strategy_cls(cls):
        TRAINING_STRATEGY_MAPPING.register(name, cls)
        return cls

    return register_training_strategy_cls


__all__ = [
    "TRAINING_STRATEGY_MAPPING",
    "get_training_strategy",
    "register_training_strategy",
]
