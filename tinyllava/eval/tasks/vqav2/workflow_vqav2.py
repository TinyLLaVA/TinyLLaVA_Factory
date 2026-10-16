"""Execution steps for the vqav2 benchmark."""

from pathlib import Path

from tinyllava.eval.tasks.workflow_base import EvaluationWorkflow


def build_workflow(data: Path, output: Path, answers: Path) -> list:
    workflow = EvaluationWorkflow("vqav2", output, answers)
    result = workflow.steps
    command = workflow.command
    base = workflow.evaluation_command
    converted = str(output / "predictions.json")
    command(
        "convert",
        base
        + [
            "--split-file",
            str(data / "vqav2/llava_vqav2_mscoco_test2015.jsonl"),
            "--output-file",
            converted,
        ],
    )
    return result
