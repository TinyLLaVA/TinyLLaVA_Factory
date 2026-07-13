"""Base protocol for dataset-specific evaluation adapters."""

from __future__ import annotations

import argparse
from typing import Any, Protocol


class DatasetEvaluation(Protocol):
    """Convert generic generation JSONL into submissions or local scores."""

    name: str
    help: str

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Register dataset-specific CLI arguments."""

    def run(self, args: argparse.Namespace) -> Any:
        """Run conversion or local evaluation and print a concise report."""
