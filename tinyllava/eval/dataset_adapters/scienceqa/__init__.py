"""ScienceQA dataset adapters."""

from tinyllava.eval.dataset_adapters.scienceqa.evaluation_scienceqa import (
    ScienceQaEvaluation,
    evaluate_scienceqa,
)
from tinyllava.eval.dataset_adapters.scienceqa.loader_scienceqa import (
    ScienceQaLoader,
)

__all__ = ["ScienceQaEvaluation", "ScienceQaLoader", "evaluate_scienceqa"]
