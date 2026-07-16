"""MMMU submission conversion."""

from __future__ import annotations

import argparse

from tinyllava.eval.utils import read_jsonl, write_json


def convert_mmmu_submission(*, prediction_file: str, output_file: str) -> int:
    """Write MMMU's question-id to answer mapping."""

    submission = {row["question_id"]: row["text"] for row in read_jsonl(prediction_file)}
    write_json(output_file, submission)
    return len(submission)


class MmmuEvaluation:
    name = "mmmu"
    help = "Convert MMMU predictions."

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--prediction-file", required=True)
        parser.add_argument("--output-file", required=True)

    def run(self, args: argparse.Namespace) -> int:
        count = convert_mmmu_submission(
            prediction_file=args.prediction_file,
            output_file=args.output_file,
        )
        print(f"MMMU submission: {count} predictions")
        return count
