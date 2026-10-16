"""Execution steps for the scienceqa benchmark."""

from pathlib import Path

from tinyllava.eval.tasks.workflow_base import EvaluationWorkflow


def build_workflow(data: Path, output: Path, answers: Path) -> list:
    workflow = EvaluationWorkflow("scienceqa", output, answers)
    result = workflow.steps
    command = workflow.command
    base = workflow.evaluation_command
    command(
        "score",
        base
        + [
            "--base-dir",
            str(data / "scienceqa"),
            "--output-file",
            str(output / "predictions.jsonl"),
            "--output-result-file",
            str(output / "metrics.json"),
        ],
    )
    return result
