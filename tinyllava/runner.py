"""Process launching and shard merging for resolved run plans."""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any


def devices(value: int | str | list[int | str]) -> list[str]:
    if isinstance(value, int):
        value = [value]
    if value == "auto":
        value = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    if isinstance(value, str):
        value = value.split(",")
    if (
        not isinstance(value, list)
        or not value
        or any(not str(v).strip() for v in value)
    ):
        raise ValueError("devices must be a nonempty list or comma-separated GPU IDs")
    result = [str(v).strip() for v in value]
    if len(set(result)) != len(result) or "-1" in result:
        raise ValueError("devices must contain distinct GPU IDs")
    return result


def run_command(
    args: Sequence[str | int | Path],
    *,
    cwd: str | Path | None = None,
    env: dict[str, str] | None = None,
    dry_run: bool = False,
) -> None:
    command = [sys.executable if str(arg) == "{python}" else str(arg) for arg in args]
    prefix = f"(cd {shlex.quote(str(cwd))}) " if cwd else ""
    if env is not None and "CUDA_VISIBLE_DEVICES" in env:
        prefix += f"CUDA_VISIBLE_DEVICES={shlex.quote(env['CUDA_VISIBLE_DEVICES'])} "
    print(prefix + shlex.join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=cwd, env=env, check=True)


def merge_shards(paths: Sequence[str | Path], destination: str | Path) -> None:
    """Replace the merged output only after every shard is readable."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent, delete=False
        ) as output:
            temporary = Path(output.name)
            for path in paths:
                with Path(path).open("rb") as source:
                    shutil.copyfileobj(source, output)
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def run_shards(commands: Sequence[list[str]], envs: Sequence[dict[str, str]]) -> None:
    processes = []
    try:
        for command, env in zip(commands, envs, strict=True):
            print(shlex.join(command), flush=True)
            processes.append(subprocess.Popen(command, env=env))
        for process in processes:
            code = process.wait()
            if code:
                raise subprocess.CalledProcessError(code, process.args)
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def run_step(
    step: dict[str, Any],
    *,
    dry_run: bool = False,
) -> None:
    """Execute a generic command or filesystem preparation step."""
    kind = step["kind"]
    if kind == "workspace":
        print(f"Prepare workspace: {step}", flush=True)
        if not dry_run:
            for directory in step.get("mkdir", []):
                Path(directory).mkdir(parents=True, exist_ok=True)
            for destination, source in step.get("symlinks", {}).items():
                link = Path(destination)
                if not link.exists():
                    link.symlink_to(source, target_is_directory=True)
            for destination, source in step.get("copies", {}).items():
                shutil.copyfile(source, destination)
        return
    if kind != "command":
        raise ValueError(f"Unknown step kind: {kind}")
    if not dry_run:
        for path in step.get("mkdir", []):
            Path(path).mkdir(parents=True, exist_ok=True)
    run_command(step["args"], cwd=step.get("cwd"), dry_run=dry_run)


def run_plan(
    plan: list[tuple[str, dict[str, Any], dict[str, Any] | None]],
    *,
    steps: Sequence[str] | None = None,
    dry_run: bool = False,
    execute_step: Callable[..., None] | None = None,
) -> None:
    """Run selected stages in order, stopping immediately on failure.

    Domain executors receive (step, config, temporary_directory, dry_run=...).
    Without an executor, stages are generic commands or workspace operations.
    """
    names = [name for name, _, _ in plan]
    selected = names if steps is None else steps
    unknown = set(selected) - set(names)
    if unknown or not selected:
        raise ValueError(
            f"Unknown or empty stage selection: {selected}; available: {names}"
        )
    by_name = {name: (step, config) for name, step, config in plan}
    for name in selected:
        step, config = by_name[name]
        print(f"Stage: {name}", flush=True)
        if execute_step is None:
            run_step(step, dry_run=dry_run)
        else:
            with tempfile.TemporaryDirectory(prefix="tinyllava-run-") as directory:
                execute_step(step, config, directory, dry_run=dry_run)
