"""Execution steps for the gqa benchmark."""

from pathlib import Path

from tinyllava.eval.tasks.workflow_base import EvaluationWorkflow


def build_workflow(data: Path, output: Path, answers: Path) -> list:
    workflow = EvaluationWorkflow("gqa", output, answers)
    result = workflow.steps
    command = workflow.command
    base = workflow.evaluation_command
    converted = str(output / "predictions.json")
    command("convert", base + ["--output-file", converted])
    command(
        "score",
        [
            "{python}",
            str(data / "gqa/eval/eval.py"),
            "--tier",
            "testdev_balanced",
            "--predictions",
            converted,
        ],
        cwd=data / "gqa",
    )
    return result
