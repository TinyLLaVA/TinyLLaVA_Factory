"""VQA dataset adapters."""

from tinyllava.eval.dataset_adapters.vqa.loader_vqa import VqaLoader
from tinyllava.eval.dataset_adapters.vqa.vqa_answer import (
    EvalAIAnswerProcessor,
    TextVQAAccuracyEvaluator,
)

__all__ = ["EvalAIAnswerProcessor", "TextVQAAccuracyEvaluator", "VqaLoader"]
