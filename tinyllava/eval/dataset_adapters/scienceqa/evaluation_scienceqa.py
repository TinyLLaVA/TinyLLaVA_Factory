"""ScienceQA multiple-choice local evaluation."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections.abc import Sequence
from typing import Any

from tinyllava.eval.utils import read_jsonl, write_json


def evaluate_scienceqa(
    *,
    base_dir: str,
    prediction_file: str,
    output_file: str,
    output_result_file: str,
    split: str = "test",
    options: Sequence[str] = ("A", "B", "C", "D", "E"),
) -> dict[str, Any]:
    """Evaluate ScienceQA multiple-choice predictions.

    The local ScienceQA conversion stores canonical split/problem metadata under
    `base_dir`; generated JSONL predictions are matched by problem id.
    """

    split_indices = json.load(open(os.path.join(base_dir, "pid_splits.json")))[split]
    problems = json.load(open(os.path.join(base_dir, "problems.json")))
    predictions = {row["question_id"]: row for row in read_jsonl(prediction_file)}
    split_problems = {idx: problems[idx] for idx in split_indices}

    detailed = {"correct": [], "incorrect": []}
    result = {"acc": None, "correct": None, "count": None, "results": {}, "outputs": {}}

    for problem_id, problem in split_problems.items():
        prediction = predictions.get(problem_id, {"text": "FAILED", "prompt": "Unknown"})
        pred_text = prediction["text"]
        answer = _parse_scienceqa_answer(pred_text, options)
        pred_idx = _choice_index(answer, problem["choices"], options)

        metadata = prediction.get("metadata", {})
        is_multimodal = metadata.get("has_image")
        if is_multimodal is None:
            is_multimodal = "<image>" in prediction["prompt"]

        analysis = {
            "question_id": problem_id,
            "parsed_ans": answer,
            "ground_truth": options[problem["answer"]],
            "question": prediction["prompt"],
            "pred": pred_text,
            "is_multimodal": bool(is_multimodal),
        }

        result["results"][problem_id] = pred_idx
        result["outputs"][problem_id] = pred_text

        bucket = "correct" if pred_idx == problem["answer"] else "incorrect"
        detailed[bucket].append(analysis)

    correct = len(detailed["correct"])
    total = correct + len(detailed["incorrect"])
    multimodal_correct = len([item for item in detailed["correct"] if item["is_multimodal"]])
    multimodal_incorrect = len(
        [item for item in detailed["incorrect"] if item["is_multimodal"]]
    )
    multimodal_total = multimodal_correct + multimodal_incorrect

    result["acc"] = correct / total * 100 if total else 0.0
    result["correct"] = correct
    result["count"] = total
    result["multimodal_acc"] = (
        multimodal_correct / multimodal_total * 100 if multimodal_total else None
    )

    write_json(output_file, detailed, indent=2)
    write_json(output_result_file, result, indent=2)
    return result


class ScienceQaEvaluation:
    name = "scienceqa"
    help = "Evaluate ScienceQA predictions."

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--base-dir", required=True)
        parser.add_argument("--prediction-file", required=True)
        parser.add_argument("--output-file", required=True)
        parser.add_argument("--output-result-file", required=True)
        parser.add_argument("--split", default="test")
        parser.add_argument("--options", nargs="+", default=["A", "B", "C", "D", "E"])

    def run(self, args: argparse.Namespace) -> dict[str, Any]:
        report = evaluate_scienceqa(
            base_dir=args.base_dir,
            prediction_file=args.prediction_file,
            output_file=args.output_file,
            output_result_file=args.output_result_file,
            split=args.split,
            options=args.options,
        )
        message = (
            f"ScienceQA: total {report['count']}, correct {report['correct']}, "
            f"accuracy {report['acc']:.2f}%"
        )
        if report["multimodal_acc"] is not None:
            message += f", IMG-Accuracy {report['multimodal_acc']:.2f}%"
        print(message)
        return report


def _parse_scienceqa_answer(prediction: str, options: Sequence[str]) -> str:
    if prediction in options:
        return prediction
    if len(prediction) >= 3 and prediction[0] in options and prediction[1:3] == ". ":
        return prediction[0]
    answers = re.compile(r"The answer is ([A-Z]).").findall(prediction)
    if len(answers) == 1:
        return answers[0]
    return "FAILED"


def _choice_index(answer: str, choices: Sequence[Any], options: Sequence[str]) -> int:
    if answer in options[: len(choices)]:
        return list(options).index(answer)
    return -1
