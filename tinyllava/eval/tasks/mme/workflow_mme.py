"""Execution steps for the mme benchmark."""

from pathlib import Path

from tinyllava.eval.tasks.workflow_base import EvaluationWorkflow


def build_workflow(data: Path, output: Path, answers: Path) -> list:
    workflow = EvaluationWorkflow("mme", output, answers)
    result = workflow.steps
    command = workflow.command
    workspace = output / "mme"
    result.append(
        (
            "prepare",
            {
                "kind": "workspace",
                "mkdir": [str(workspace / "answers")],
                "symlinks": {
                    str(workspace / "MME_Benchmark_release_version"): str(
                        data / "MME/MME_Benchmark_release_version"
                    )
                },
                "copies": {str(workspace / "answers/model.jsonl"): str(answers)},
            },
            None,
        )
    )
    command(
        "convert",
        [
            "{python}",
            str(data / "MME/convert_answer_to_mme.py"),
            "--experiment",
            "model",
        ],
        cwd=workspace,
    )
    command(
        "score",
        [
            "{python}",
            str(data / "MME/eval_tool/calculation.py"),
            "--results_dir",
            str(workspace / "eval_tool/answers/model"),
        ],
    )
    return result
