"""Training sample adapter interface."""

from collections.abc import Mapping
from typing import Any, Protocol

from datasets import Features


class TrainingDatasetAdapter(Protocol):
    """Project source-specific rows into a stable training schema."""

    name: str
    cache_version: str
    features: Features

    def adapt(self, sample: Mapping[str, Any]) -> dict[str, Any]:
        """Convert one raw source row into the canonical schema."""


__all__ = ["TrainingDatasetAdapter"]
