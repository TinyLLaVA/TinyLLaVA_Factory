"""Execution steps for the textvqa benchmark."""

from pathlib import Path

from tinyllava.eval.tasks.workflow_base import EvaluationWorkflow


def build_workflow(data: Path, output: Path, answers: Path) -> list:
    workflow = EvaluationWorkflow("textvqa", output, answers)
    result = workflow.steps
    command = workflow.command
    base = workflow.evaluation_command
    command(
        "score",
        base + ["--annotation-file", str(data / "textvqa/TextVQA_0.5.1_val.json")],
    )
    return result
