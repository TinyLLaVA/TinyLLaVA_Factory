"""Base types for dataset loader adapters."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class GenerationExample:
    question_id: str
    prompt: str
    image_files: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class DatasetLoader(Protocol):
    """Convert benchmark samples into processor-ready generation examples."""

    def load_samples(self, question_file: str) -> list[Mapping[str, Any]]:
        """Load benchmark questions from the dataset's local annotation format."""

    def make_example(
        self,
        sample: Mapping[str, Any],
        *,
        image_folder: str,
        args: Any,
    ) -> GenerationExample:
        """Convert one loaded sample into prompt text plus image paths."""

    def process_response(self, sample: Mapping[str, Any], response: str) -> str:
        """Normalize one raw generation before writing the generic JSONL output."""
