"""GQA evaluation adapter."""

from tinyllava.eval.dataset_adapters.gqa.evaluation_gqa import (
    GqaEvaluation,
    convert_gqa_submission,
)

__all__ = ["GqaEvaluation", "convert_gqa_submission"]
