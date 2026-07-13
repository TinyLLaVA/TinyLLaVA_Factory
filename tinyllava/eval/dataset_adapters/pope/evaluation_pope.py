"""POPE hallucination benchmark local evaluation."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from tinyllava.eval.utils import read_jsonl


@dataclass(frozen=True)
class PopeScores:
    category: str
    samples: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    yes_ratio: float
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int


def evaluate_pope(
    *,
    annotation_dir: str,
    question_file: str,
    prediction_file: str,
) -> list[PopeScores]:
    """Evaluate POPE hallucination splits.

    POPE labels are binary yes/no. The standard LLaVA script maps free-form
    generations to yes/no by inspecting the first sentence; we preserve that
    behavior so old POPE reports stay comparable.
    """

    questions = {row["question_id"]: row for row in read_jsonl(question_file)}
    predictions = read_jsonl(prediction_file)
    scores = []

    for filename in sorted(os.listdir(annotation_dir)):
        if not filename.startswith("coco_pope_") or not filename.endswith(".json"):
            continue
        category = filename[len("coco_pope_") : -len(".json")]
        category_predictions = [
            row
            for row in predictions
            if questions[row["question_id"]]["category"] == category
        ]
        label_file = os.path.join(annotation_dir, filename)
        scores.append(_score_pope_category(category, category_predictions, label_file))

    return scores


class PopeEvaluation:
    name = "pope"
    help = "Evaluate POPE predictions."

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--annotation-dir", required=True)
        parser.add_argument("--question-file", required=True)
        parser.add_argument("--prediction-file", required=True)

    def run(self, args: argparse.Namespace) -> list[PopeScores]:
        scores = evaluate_pope(
            annotation_dir=args.annotation_dir,
            question_file=args.question_file,
            prediction_file=args.prediction_file,
        )
        for score in scores:
            print(f"Category: {score.category}, # samples: {score.samples}")
            print("TP\tFP\tTN\tFN")
            print(
                f"{score.true_positive}\t{score.false_positive}\t"
                f"{score.true_negative}\t{score.false_negative}"
            )
            print(f"Accuracy: {score.accuracy}")
            print(f"Precision: {score.precision}")
            print(f"Recall: {score.recall}")
            print(f"F1 score: {score.f1}")
            print(f"Yes ratio: {score.yes_ratio}")
            print(
                "%.3f, %.3f, %.3f, %.3f, %.3f"
                % (
                    score.f1,
                    score.accuracy,
                    score.precision,
                    score.recall,
                    score.yes_ratio,
                )
            )
            print("====================================")
        return scores


def _score_pope_category(
    category: str,
    predictions: Iterable[Mapping[str, Any]],
    label_file: str,
) -> PopeScores:
    labels = [json.loads(line)["label"] for line in open(label_file, encoding="utf-8")]
    label_values = [0 if label == "no" else 1 for label in labels]
    pred_values = [_pope_yes_no(row["text"]) for row in predictions]
    yes_ratio = pred_values.count(1) / len(pred_values) if pred_values else 0.0

    true_positive = true_negative = false_positive = false_negative = 0
    for pred, label in zip(pred_values, label_values):
        if pred == 1 and label == 1:
            true_positive += 1
        elif pred == 1 and label == 0:
            false_positive += 1
        elif pred == 0 and label == 0:
            true_negative += 1
        elif pred == 0 and label == 1:
            false_negative += 1

    eps = 1e-6
    precision = true_positive / (true_positive + false_positive + eps)
    recall = true_positive / (true_positive + false_negative + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    total = true_positive + true_negative + false_positive + false_negative
    accuracy = (true_positive + true_negative) / total if total else 0.0

    return PopeScores(
        category=category,
        samples=len(pred_values),
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        yes_ratio=yes_ratio,
        true_positive=true_positive,
        false_positive=false_positive,
        true_negative=true_negative,
        false_negative=false_negative,
    )


def _pope_yes_no(text: str) -> int:
    first_sentence = text.split(".", 1)[0]
    words = first_sentence.replace(",", "").split(" ")
    if "No" in words or "not" in words or "no" in words:
        return 0
    return 1
