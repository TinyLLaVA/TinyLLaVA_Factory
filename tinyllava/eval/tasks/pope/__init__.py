"""POPE dataset adapters."""

from tinyllava.eval.tasks.pope.evaluation_pope import (
    PopeEvaluation,
    PopeScores,
    evaluate_pope,
)
from tinyllava.eval.tasks.pope.loader_pope import PopeLoader

__all__ = ["PopeEvaluation", "PopeLoader", "PopeScores", "evaluate_pope"]
