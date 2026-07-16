"""VQAv2 evaluation adapter."""

from tinyllava.eval.tasks.vqav2.evaluation_vqav2 import (
    Vqav2Evaluation,
    convert_vqav2_submission,
)

__all__ = ["Vqav2Evaluation", "convert_vqav2_submission"]
