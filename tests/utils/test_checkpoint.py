from types import SimpleNamespace

import pytest

from tinyllava.utils.checkpoint import (
    find_last_complete_checkpoint,
    resolve_resume_checkpoint,
)


def _write_checkpoint(root, step: int, *, deepspeed: bool = False):
    checkpoint = root / f"checkpoint-{step}"
    checkpoint.mkdir(parents=True)
    (checkpoint / "trainer_state.json").write_text("{}", encoding="utf-8")
    if deepspeed:
        (checkpoint / "latest").write_text(f"global_step{step}", encoding="utf-8")
        (checkpoint / f"global_step{step}").mkdir()
    return checkpoint


def test_find_last_complete_checkpoint_skips_partial_newer_directory(tmp_path):
    expected = _write_checkpoint(tmp_path, 500, deepspeed=True)
    partial = tmp_path / "checkpoint-1000"
    partial.mkdir()
    (partial / "trainer_state.json").write_text("{}", encoding="utf-8")

    assert find_last_complete_checkpoint(str(tmp_path), deepspeed=True) == str(expected)


def test_auto_resume_starts_fresh_when_output_has_no_checkpoint(tmp_path):
    args = SimpleNamespace(
        resume_from_checkpoint="auto",
        output_dir=str(tmp_path),
        deepspeed=None,
    )

    assert resolve_resume_checkpoint(args) is None


def test_explicit_resume_rejects_incomplete_checkpoint(tmp_path):
    checkpoint = tmp_path / "checkpoint-500"
    checkpoint.mkdir()
    args = SimpleNamespace(
        resume_from_checkpoint=str(checkpoint),
        output_dir=str(tmp_path),
        deepspeed=None,
    )

    with pytest.raises(ValueError, match="incomplete or invalid"):
        resolve_resume_checkpoint(args)
