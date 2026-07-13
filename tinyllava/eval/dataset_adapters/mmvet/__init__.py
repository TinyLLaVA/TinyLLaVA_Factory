"""MM-Vet evaluation adapter."""

from tinyllava.eval.dataset_adapters.mmvet.evaluation_mmvet import (
    MmvetEvaluation,
    convert_mmvet_submission,
)

__all__ = ["MmvetEvaluation", "convert_mmvet_submission"]
