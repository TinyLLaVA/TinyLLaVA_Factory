"""ScienceQA dataset adapters."""

from tinyllava.eval.tasks.scienceqa.evaluation_scienceqa import (
    ScienceQaEvaluation,
    evaluate_scienceqa,
)
from tinyllava.eval.tasks.scienceqa.loader_scienceqa import (
    ScienceQaLoader,
)

__all__ = ["ScienceQaEvaluation", "ScienceQaLoader", "evaluate_scienceqa"]
