"""MM-Vet submission conversion."""

from __future__ import annotations

import argparse

from tinyllava.eval.utils import read_jsonl, write_json


def convert_mmvet_submission(*, prediction_file: str, output_file: str) -> int:
    """Write MM-Vet's `v1_<question_id>` answer mapping."""

    submission = {
        f"v1_{row['question_id']}": row["text"] for row in read_jsonl(prediction_file)
    }
    write_json(output_file, submission, indent=2)
    return len(submission)


class MmvetEvaluation:
    name = "mmvet"
    help = "Convert MM-Vet predictions."

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--prediction-file", required=True)
        parser.add_argument("--output-file", required=True)

    def run(self, args: argparse.Namespace) -> int:
        count = convert_mmvet_submission(
            prediction_file=args.prediction_file,
            output_file=args.output_file,
        )
        print(f"MM-Vet submission: {count} predictions")
        return count
