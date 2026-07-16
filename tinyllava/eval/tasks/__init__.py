"""Dataset-specific loader and evaluation tasks."""

from tinyllava.eval.tasks.loader_base import (
    DatasetLoader,
    GenerationExample,
)
from tinyllava.eval.tasks.evaluation_base import DatasetEvaluation

__all__ = [
    "DatasetEvaluation",
    "DatasetLoader",
    "GenerationExample",
]
