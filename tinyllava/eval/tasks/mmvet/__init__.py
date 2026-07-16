"""MM-Vet evaluation adapter."""

from tinyllava.eval.tasks.mmvet.evaluation_mmvet import (
    MmvetEvaluation,
    convert_mmvet_submission,
)

__all__ = ["MmvetEvaluation", "convert_mmvet_submission"]
