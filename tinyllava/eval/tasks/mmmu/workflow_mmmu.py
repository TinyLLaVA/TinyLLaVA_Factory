"""Execution steps for the mmmu benchmark."""

from pathlib import Path

from tinyllava.eval.tasks.workflow_base import EvaluationWorkflow


def build_workflow(data: Path, output: Path, answers: Path) -> list:
    workflow = EvaluationWorkflow("mmmu", output, answers)
    result = workflow.steps
    command = workflow.command
    base = workflow.evaluation_command
    converted = str(output / "predictions.json")
    command("convert", base + ["--output-file", converted])
    command(
        "score",
        [
            "{python}",
            str(data / "MMMU/eval/main_eval_only.py"),
            "--output_path",
            converted,
        ],
        cwd=data / "MMMU/eval",
    )
    return result
