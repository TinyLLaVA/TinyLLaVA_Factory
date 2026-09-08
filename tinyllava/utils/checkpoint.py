"""Checkpoint discovery for safe Trainer/DeepSpeed resume."""

import re
from pathlib import Path

from transformers import TrainingArguments

from .logging import log


_CHECKPOINT_PATTERN = re.compile(r"^checkpoint-(\d+)$")


def _is_complete_checkpoint(path: Path, *, deepspeed: bool) -> bool:
    if not (path / "trainer_state.json").is_file():
        return False
    if not deepspeed:
        return True

    latest = path / "latest"
    if not latest.is_file():
        return False
    tag = latest.read_text(encoding="utf-8").strip()
    return bool(tag) and (path / tag).is_dir()


def find_last_complete_checkpoint(
    output_dir: str,
    *,
    deepspeed: bool = False,
) -> str | None:
    """Find the highest-numbered checkpoint with the required resume markers.

    A candidate must contain `trainer_state.json`. With DeepSpeed, it must also
    contain a nonempty `latest` file pointing to an existing state directory.
    These checks verify resume markers, not the integrity of every weight shard.

    Args:
        output_dir: Directory containing `checkpoint-<step>` subdirectories.
        deepspeed: Require the DeepSpeed state markers in addition to Trainer state.

    Returns:
        The selected checkpoint path, or `None` if no candidate passes the checks.
    """
    root = Path(output_dir).expanduser()
    if not root.is_dir():
        return None

    candidates: list[tuple[int, Path]] = []
    for child in root.iterdir():
        match = _CHECKPOINT_PATTERN.match(child.name)
        if match and child.is_dir():
            candidates.append((int(match.group(1)), child))

    for _, checkpoint in sorted(candidates, reverse=True):
        if _is_complete_checkpoint(checkpoint, deepspeed=deepspeed):
            return str(checkpoint)
    return None


def resolve_resume_checkpoint(training_args: TrainingArguments) -> str | None:
    """Resolve a resume request to a checkpoint path.

    Args:
        training_args: Trainer settings containing the resume request, output
            directory, and DeepSpeed configuration. `auto` and `latest` select the
            newest valid candidate; `None` and `False` disable resume.

    Returns:
        A checkpoint path, or `None` to start a new run. Automatic discovery falls
        back to a new run when no candidate exists.

    Raises:
        ValueError: An explicit path fails the resume-marker checks, or the request
            is `True` and automatic discovery finds no checkpoint.
    """
    requested = training_args.resume_from_checkpoint
    if requested in (None, False):
        return None

    uses_deepspeed = bool(training_args.deepspeed)
    if requested is True or str(requested).lower() in {"auto", "latest"}:
        checkpoint = find_last_complete_checkpoint(
            training_args.output_dir,
            deepspeed=uses_deepspeed,
        )
        if checkpoint is None:
            if requested is True:
                raise ValueError(
                    "resume_from_checkpoint=true, but no complete checkpoint "
                    f"was found under {training_args.output_dir!r}."
                )
            log(
                "No complete checkpoint found under "
                f"{training_args.output_dir}; starting a new run."
            )
            return None
    else:
        checkpoint_path = Path(str(requested)).expanduser()
        if not _is_complete_checkpoint(
            checkpoint_path,
            deepspeed=uses_deepspeed,
        ):
            raise ValueError(
                f"Checkpoint {str(checkpoint_path)!r} is incomplete or invalid."
            )
        checkpoint = str(checkpoint_path)

    log(f"Resuming training from complete checkpoint {checkpoint}.")
    return checkpoint


__all__ = ["find_last_complete_checkpoint", "resolve_resume_checkpoint"]
