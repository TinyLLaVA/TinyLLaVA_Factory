"""Helpers for benchmark-owned conversion and scoring workflows."""

from pathlib import Path


class EvaluationWorkflow:
    """Collect commands; constructing a workflow performs no filesystem writes."""

    def __init__(self, benchmark: str, output: Path, answers: Path):
        self.steps = []
        self.output = output
        self.evaluation_command = [
            "{python}",
            "-m",
            "tinyllava.eval.tasks.auto.evaluation_auto",
            benchmark,
            "--prediction-file",
            str(answers),
        ]

    def command(
        self, name: str, args: list[str], cwd: str | Path | None = None
    ) -> None:
        self.steps.append(
            (
                name,
                {
                    "kind": "command",
                    "args": args,
                    "cwd": str(cwd) if cwd else None,
                    "mkdir": [str(self.output)],
                },
                None,
            )
        )
