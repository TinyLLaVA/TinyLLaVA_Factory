"""Base protocol for dataset-specific evaluation adapters."""

from __future__ import annotations

import argparse
from typing import Any, Protocol


class DatasetEvaluation(Protocol):
    """Convert generic prediction JSONL into benchmark submissions or local scores.

    Attributes:
        name: Evaluation subcommand name.
        help: Short description displayed by the command-line parser.
    """

    name: str
    help: str

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Register benchmark-specific options on a command-line parser.

        Args:
            parser: Parser to extend in place with input, output, and scoring options.
        """

    def run(self, args: argparse.Namespace) -> Any:
        """Export predictions or compute local benchmark scores.

        Args:
            args: Parsed options registered by `add_arguments`.

        Returns:
            Adapter-defined results, such as metrics or an export path.
        """
