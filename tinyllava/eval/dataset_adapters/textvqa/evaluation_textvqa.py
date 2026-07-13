"""TextVQA local accuracy evaluation."""

from __future__ import annotations

import argparse
import json
import re

from tinyllava.eval.utils import read_jsonl
from tinyllava.eval.dataset_adapters.vqa.vqa_answer import TextVQAAccuracyEvaluator


def evaluate_textvqa(*, annotation_file: str, prediction_file: str) -> dict[str, float]:
    """Evaluate TextVQA predictions with the M4C/TextVQA accuracy rule."""

    annotations = json.load(open(annotation_file, encoding="utf-8"))["data"]
    annotations = {
        (annotation["image_id"], _normalize_textvqa_question(annotation["question"])):
        annotation
        for annotation in annotations
    }

    pred_list = []
    for row in read_jsonl(prediction_file):
        key = (row["question_id"], _extract_textvqa_question(row["prompt"]))
        annotation = annotations[key]
        pred_list.append({"pred_answer": row["text"], "gt_answers": annotation["answers"]})

    accuracy = TextVQAAccuracyEvaluator().eval_pred_list(pred_list)
    return {"samples": len(pred_list), "accuracy": accuracy}


class TextVqaEvaluation:
    name = "textvqa"
    help = "Evaluate TextVQA predictions."

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--annotation-file", required=True)
        parser.add_argument("--prediction-file", required=True)

    def run(self, args: argparse.Namespace) -> dict[str, float]:
        report = evaluate_textvqa(
            annotation_file=args.annotation_file,
            prediction_file=args.prediction_file,
        )
        print(
            f"TextVQA: {report['samples']} samples, "
            f"accuracy {100.0 * report['accuracy']:.2f}%"
        )
        return report


def _normalize_textvqa_question(question: str) -> str:
    return question.lower()


def _extract_textvqa_question(prompt: str) -> str:
    if prompt.startswith("OCR tokens: "):
        pattern = r"Question: (.*?) Short answer:"
        match = re.search(pattern, prompt, re.DOTALL)
        if match is None:
            raise ValueError(f"Cannot parse TextVQA prompt: {prompt!r}")
        question = match.group(1)
    elif "Reference OCR token: " in prompt and len(prompt.split("\n")) == 3:
        if prompt.startswith("Reference OCR token:"):
            question = prompt.split("\n")[1]
        else:
            question = prompt.split("\n")[0]
    elif len(prompt.split("\n")) == 2:
        question = prompt.split("\n")[0]
    else:
        raise ValueError(f"Cannot parse TextVQA prompt: {prompt!r}")
    return _normalize_textvqa_question(question)
