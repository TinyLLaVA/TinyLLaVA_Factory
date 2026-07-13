"""GQA official evaluation submission conversion."""

from __future__ import annotations

import argparse

from tinyllava.eval.utils import read_jsonl, write_json


def convert_gqa_submission(*, prediction_file: str, output_file: str) -> int:
    """Write GQA's official `questionId`/`prediction` JSON format."""

    submission = []
    for row in read_jsonl(prediction_file):
        submission.append(
            {
                "questionId": row["question_id"],
                "prediction": row["text"].rstrip(".").lower(),
            }
        )
    write_json(output_file, submission)
    return len(submission)


class GqaEvaluation:
    name = "gqa"
    help = "Convert GQA predictions for official eval."

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--prediction-file", required=True)
        parser.add_argument("--output-file", required=True)

    def run(self, args: argparse.Namespace) -> int:
        count = convert_gqa_submission(
            prediction_file=args.prediction_file,
            output_file=args.output_file,
        )
        print(f"GQA submission: {count} predictions")
        return count
