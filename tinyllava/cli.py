"""Unified CLI: python -m tinyllava.cli {train,eval,single-turn,chat}."""

from __future__ import annotations

import argparse
from collections.abc import Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, description in (
        ("train", "Run a training stage or multi-stage experiment."),
        ("eval", "Generate predictions and run benchmark conversion/scoring."),
    ):
        command = commands.add_parser(name, help=description)
        command.add_argument("--config", required=True)
        command.add_argument(
            "--steps", help="Comma-separated stage names in execution order."
        )
        command.add_argument("--dry-run", action="store_true")
        command.add_argument(
            "overrides",
            nargs="*",
            help="Config overrides: key=value, model=phi, or section.from=path.yaml.",
        )

    for name, description in (
        ("single-turn", "Answer one prompt and exit."),
        ("chat", "Interactive conversation with retained history."),
    ):
        command = commands.add_parser(name, help=description)
        command.add_argument("--model-path", required=True)
        command.add_argument("--query", required=name == "single-turn")
        command.add_argument("--image-file", action="append", dest="image_files")
        command.add_argument(
            "--device", help="Default: CUDA if available, otherwise CPU."
        )
        command.add_argument("--chat-template-path")
        command.add_argument("--temperature", type=float, default=0.2)
        command.add_argument("--top-p", type=float)
        command.add_argument("--num-beams", type=int, default=1)
        command.add_argument("--max-new-tokens", type=int, default=512)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command in {"train", "eval"}:
        if args.command == "train":
            from tinyllava.train.workflow import run
        else:
            from tinyllava.eval.workflow import run

        run(
            args.config,
            args.overrides,
            steps=args.steps.split(",") if args.steps else None,
            dry_run=args.dry_run,
        )
    else:
        from tinyllava.serve.chat import run_chat

        options = vars(args).copy()
        command = options.pop("command")
        run_chat(**options, interactive=command == "chat")


if __name__ == "__main__":
    main()
