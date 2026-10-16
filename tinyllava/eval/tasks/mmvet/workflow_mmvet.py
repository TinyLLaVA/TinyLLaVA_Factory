"""Execution steps for the mmvet benchmark."""

from pathlib import Path

from tinyllava.eval.tasks.workflow_base import EvaluationWorkflow


def build_workflow(data: Path, output: Path, answers: Path) -> list:
    workflow = EvaluationWorkflow("mmvet", output, answers)
    result = workflow.steps
    command = workflow.command
    base = workflow.evaluation_command
    converted = str(output / "predictions.json")
    command("convert", base + ["--output-file", converted])
    return result
