import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from tinyllava.cli import main

ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize("command", ["train", "eval"])
def test_explicit_command_selects_plan(command):
    with patch(f"tinyllava.{command}.workflow.run") as run:
        main(
            [
                command,
                "--config",
                "config.yaml",
                "--steps",
                "one,two",
                "--dry-run",
                "model.from=configs/models/phi.yaml",
            ]
        )
    run.assert_called_once_with(
        "config.yaml",
        ["model.from=configs/models/phi.yaml"],
        steps=["one", "two"],
        dry_run=True,
    )


@pytest.mark.parametrize("command", ["single-turn", "chat"])
def test_conversation_dispatch(command):
    with patch("tinyllava.serve.chat.run_chat") as run:
        main(
            [
                command,
                "--model-path",
                "checkpoint",
                "--query",
                "hello",
                "--image-file",
                "one.png",
            ]
        )
    assert run.call_args.kwargs["interactive"] == (command == "chat")
    assert run.call_args.kwargs["image_files"] == ["one.png"]
    assert run.call_args.kwargs["query"] == "hello"


def test_help_does_not_load_model_dependencies():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from tinyllava.cli import build_parser; "
            "build_parser().parse_args(['chat', '--model-path', 'test']); "
            "assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_single_turn_requires_query():
    with pytest.raises(SystemExit) as error:
        main(["single-turn", "--model-path", "checkpoint"])
    assert error.value.code == 2
