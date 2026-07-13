"""Lazy auto factory and CLI for dataset evaluation adapters."""

from __future__ import annotations

import argparse
import importlib
from collections.abc import Sequence
from typing import cast

from tinyllava.eval.dataset_adapters.auto.auto_mappings import (
    DATASET_EVALUATION_MAPPING_NAMES,
)
from tinyllava.eval.dataset_adapters.evaluation_base import DatasetEvaluation


class _LazyDatasetEvaluationMapping:
    """Lazy mapping that imports dataset evaluation modules only when requested."""

    def __init__(self, mapping: dict[str, str]):
        self._mapping = mapping
        self._modules = {}
        self._extra_content = {}

    def __getitem__(self, key: str) -> type[DatasetEvaluation]:
        if key in self._extra_content:
            return cast(type[DatasetEvaluation], self._extra_content[key])
        if key not in self._mapping:
            raise KeyError(key)

        if key not in self._modules:
            self._modules[key] = importlib.import_module(
                f".{key}.evaluation_{key}", "tinyllava.eval.dataset_adapters"
            )
        return cast(
            type[DatasetEvaluation],
            getattr(self._modules[key], self._mapping[key]),
        )

    def keys(self):
        return self._mapping.keys()

    def register(self, key: str, value: type[DatasetEvaluation]) -> None:
        self._extra_content[key] = value


DATASET_EVALUATION_MAPPING = _LazyDatasetEvaluationMapping(
    DATASET_EVALUATION_MAPPING_NAMES
)


class AutoDatasetEvaluation:
    _evaluation_mapping = DATASET_EVALUATION_MAPPING

    @classmethod
    def from_name(cls, name: str) -> DatasetEvaluation:
        try:
            return cls._evaluation_mapping[name]()
        except KeyError as exc:
            raise ValueError(
                f"Unknown dataset evaluation adapter {name!r}. "
                f"Available: {sorted(cls._evaluation_mapping.keys())}"
            ) from exc

    @classmethod
    def register(cls, name: str, adapter_class: type[DatasetEvaluation]) -> None:
        cls._evaluation_mapping.register(name, adapter_class)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert generic TinyLLaVA JSONL generations into benchmark "
            "submission files, or score them with released local annotations."
        )
    )
    subparsers = parser.add_subparsers(dest="dataset", required=True)
    for name in DATASET_EVALUATION_MAPPING.keys():
        adapter = AutoDatasetEvaluation.from_name(name)
        subparser = subparsers.add_parser(name, help=adapter.help)
        adapter.add_arguments(subparser)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    AutoDatasetEvaluation.from_name(args.dataset).run(args)


if __name__ == "__main__":
    main()
