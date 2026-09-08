"""Base types for benchmark task loaders."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class GenerationExample:
    """A benchmark question ready for multimodal generation.

    Attributes:
        question_id: Identifier used to match predictions to annotations.
        prompt: User prompt passed to the processor.
        image_files: Ordered image paths or URLs; empty for text-only questions.
        metadata: Dataset-specific values preserved for prediction export.
    """

    question_id: str
    prompt: str
    image_files: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class DatasetLoader(Protocol):
    """Convert benchmark samples into processor-ready generation examples."""

    def load_samples(self, question_file: str) -> list[Mapping[str, Any]]:
        """Read questions from a benchmark annotation file.

        Args:
            question_file: Path to annotations in the loader's supported format.

        Returns:
            Sample mappings in generation order.
        """

    def make_example(
        self,
        sample: Mapping[str, Any],
        *,
        image_folder: str,
        args: Any,
    ) -> GenerationExample:
        """Convert one sample into a generation request.

        Args:
            sample: A record returned by `load_samples`.
            image_folder: Root directory for relative image paths.
            args: Runtime settings used by the dataset-specific loader.

        Returns:
            The question identifier, prompt, image paths, and export metadata.
        """

    def process_response(self, sample: Mapping[str, Any], response: str) -> str:
        """Normalize a generated answer for prediction export.

        Args:
            sample: Original annotation record.
            response: Decoded model answer.

        Returns:
            Answer text in the format expected by the benchmark.
        """
