"""VQA dataset adapters."""

from tinyllava.eval.tasks.vqa.loader_vqa import VqaLoader
from tinyllava.eval.tasks.vqa.vqa_answer import (
    EvalAIAnswerProcessor,
    TextVQAAccuracyEvaluator,
)

__all__ = ["EvalAIAnswerProcessor", "TextVQAAccuracyEvaluator", "VqaLoader"]
