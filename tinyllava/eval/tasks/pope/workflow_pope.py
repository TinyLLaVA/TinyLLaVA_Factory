"""Execution steps for the pope benchmark."""

from pathlib import Path

from tinyllava.eval.tasks.workflow_base import EvaluationWorkflow


def build_workflow(data: Path, output: Path, answers: Path) -> list:
    workflow = EvaluationWorkflow("pope", output, answers)
    result = workflow.steps
    command = workflow.command
    base = workflow.evaluation_command
    command(
        "score",
        base
        + [
            "--annotation-dir",
            str(data / "pope/coco"),
            "--question-file",
            str(data / "pope/llava_pope_test.jsonl"),
        ],
    )
    return result
