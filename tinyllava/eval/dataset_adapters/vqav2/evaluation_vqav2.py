"""VQAv2 EvalAI submission conversion."""

from __future__ import annotations

import argparse

from tinyllava.eval.utils import read_jsonl, read_jsonl_with_errors, write_json
from tinyllava.eval.dataset_adapters.vqa.vqa_answer import EvalAIAnswerProcessor


def convert_vqav2_submission(
    *,
    prediction_file: str,
    split_file: str,
    output_file: str,
) -> dict[str, int]:
    """Write VQAv2 EvalAI submission JSON.

    VQAv2 official scoring normalizes answers before comparison. Keeping
    `EvalAIAnswerProcessor` here is intentional: raw generated strings can
    differ in punctuation, articles, or number words while still being the same
    answer under the benchmark protocol.
    """

    predictions, error_lines = read_jsonl_with_errors(prediction_file)
    prediction_by_id = {item["question_id"]: item["text"] for item in predictions}
    split_rows = read_jsonl(split_file)
    answer_processor = EvalAIAnswerProcessor()

    submission = []
    missing = 0
    for row in split_rows:
        question_id = row["question_id"]
        if question_id not in prediction_by_id:
            missing += 1
            answer = ""
        else:
            answer = answer_processor(prediction_by_id[question_id])
        submission.append({"question_id": question_id, "answer": answer})

    write_json(output_file, submission)
    return {
        "predictions": len(prediction_by_id),
        "split": len(split_rows),
        "missing": missing,
        "error_lines": error_lines,
    }


class Vqav2Evaluation:
    name = "vqav2"
    help = "Convert VQAv2 predictions for EvalAI."

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--prediction-file", required=True)
        parser.add_argument("--split-file", required=True)
        parser.add_argument("--output-file", required=True)

    def run(self, args: argparse.Namespace) -> dict[str, int]:
        stats = convert_vqav2_submission(
            prediction_file=args.prediction_file,
            split_file=args.split_file,
            output_file=args.output_file,
        )
        print(
            "VQAv2 submission: "
            f"{stats['predictions']} predictions, {stats['split']} split rows, "
            f"{stats['missing']} missing, {stats['error_lines']} malformed lines"
        )
        return stats
