"""Resolve benchmark workflows without importing model or scoring dependencies."""

from importlib import import_module
from pathlib import Path

WORKFLOW_NAMES = (
    "gqa",
    "vqav2",
    "scienceqa",
    "textvqa",
    "pope",
    "mmvet",
    "mmmu",
    "mme",
)


def build_evaluation_workflow(
    benchmark: str, data: Path, output: Path, answers: Path
) -> list:
    if benchmark not in WORKFLOW_NAMES:
        raise ValueError(f"No scoring workflow for benchmark {benchmark!r}")
    module = import_module(f"tinyllava.eval.tasks.{benchmark}.workflow_{benchmark}")
    return module.build_workflow(data, output, answers)
