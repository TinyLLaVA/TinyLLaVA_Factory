import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from omegaconf import OmegaConf

from tinyllava import runner
from tinyllava.eval.workflow import eval_plan, run_generation_step
from tinyllava.runner import run_plan

ROOT = Path(__file__).parents[1]


def test_merge_preserves_existing_output_on_missing_shard(tmp_path):
    first, second, output = [
        tmp_path / name for name in ["0.jsonl", "1.jsonl", "merged.jsonl"]
    ]
    first.write_text("first\n")
    output.write_text("previous\n")
    with pytest.raises(FileNotFoundError):
        runner.merge_shards([first, second], output)
    assert output.read_text() == "previous\n"
    second.write_text("second\n")
    runner.merge_shards([first, second], output)
    assert output.read_text() == "first\nsecond\n"


def test_failed_shard_raises():
    commands = [
        [sys.executable, "-c", "pass"],
        [sys.executable, "-c", "raise SystemExit(7)"],
    ]
    with pytest.raises(subprocess.CalledProcessError) as exc:
        runner.run_shards(commands, [os.environ.copy(), os.environ.copy()])
    assert exc.value.returncode == 7


def test_sharded_generation_assigns_devices_and_merges(tmp_path, monkeypatch):
    _, step, stage = eval_plan(ROOT / "configs/eval/gqa.yaml", ["devices=[2,5]"])[0]
    stage["output"]["answers_file"] = str(tmp_path / "answers.jsonl")

    def fake_workers(commands, envs):
        for i, (command, env) in enumerate(zip(commands, envs, strict=True)):
            shard = OmegaConf.load(command[-1])
            assert env["CUDA_VISIBLE_DEVICES"] == ["2", "5"][i]
            assert shard.runtime.device == "cuda:0"
            assert shard.runtime.chunk_idx == i
            assert shard.runtime.num_chunks == 2
            Path(shard.output.answers_file).write_text(f"{i}\n")

    monkeypatch.setattr(runner, "run_shards", fake_workers)
    run_generation_step(step, stage, tmp_path)
    assert (tmp_path / "answers.jsonl").read_text() == "0\n1\n"


def test_failure_stops_later_stages(tmp_path):
    plan = [
        (
            "fail",
            {"kind": "command", "args": ["{python}", "-c", "raise SystemExit(3)"]},
            None,
        ),
        ("later", {"kind": "command", "args": ["{python}", "-c", "pass"]}, None),
    ]
    with patch(
        "tinyllava.runner.subprocess.run",
        side_effect=subprocess.CalledProcessError(3, "job"),
    ) as run:
        with pytest.raises(subprocess.CalledProcessError):
            run_plan(plan)
    assert run.call_count == 1


def test_failed_shards_leave_existing_output(tmp_path, monkeypatch):
    _, step, stage = eval_plan(ROOT / "configs/eval/gqa.yaml")[0]
    output = tmp_path / "answers.jsonl"
    output.write_text("previous\n")
    stage["output"]["answers_file"] = str(output)

    def fail(*args):
        raise subprocess.CalledProcessError(1, "worker")

    monkeypatch.setattr(runner, "run_shards", fail)
    with pytest.raises(subprocess.CalledProcessError):
        run_generation_step(step, stage, tmp_path)
    assert output.read_text() == "previous\n"


def test_mme_workspace_preparation_and_dry_run(tmp_path):
    data = tmp_path / "dataset"
    data.mkdir()
    answers = tmp_path / "answers.jsonl"
    answers.write_text('{"answer": "yes"}\n')
    workspace = tmp_path / "workspace"
    plan = [
        (
            "prepare",
            {
                "kind": "workspace",
                "mkdir": [str(workspace / "answers")],
                "symlinks": {
                    str(workspace / "MME_Benchmark_release_version"): str(data)
                },
                "copies": {str(workspace / "answers/model.jsonl"): str(answers)},
            },
            None,
        )
    ]
    run_plan(plan, dry_run=True)
    assert not workspace.exists()
    run_plan(plan)
    assert (workspace / "MME_Benchmark_release_version").resolve() == data
    assert (workspace / "answers/model.jsonl").read_text() == answers.read_text()


def test_unknown_step_kind_does_not_start_generation(tmp_path):
    with patch("tinyllava.runner.subprocess.run") as run:
        with pytest.raises(ValueError, match="Unknown step kind"):
            runner.run_step({"kind": "invalid"})
    run.assert_not_called()
