"""Process launching and shard merging for resolved run plans."""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


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


def stage_config(step: dict[str, Any]) -> dict[str, Any]:
    config = OmegaConf.merge(OmegaConf.load(step["config"]), step.get("overrides", {}))
    return OmegaConf.to_container(config, resolve=True)


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def prepare_step(step: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(step, dict):
        raise ValueError("Each workflow step must be a mapping")
    kind = step.get("kind")
    allowed = {
        "train": {
            "config",
            "overrides",
            "devices",
            "distributed",
            "master_port",
            "global_batch_size",
        },
        "generate": {"config", "overrides", "devices", "shard"},
        "command": {"args", "cwd", "mkdir"},
    }
    if kind not in allowed:
        raise ValueError(f"Unknown step kind: {kind}")
    unknown = set(step) - allowed[kind] - {"kind", "resolved"}
    if unknown:
        raise ValueError(f"Unknown {kind} step fields: {sorted(unknown)}")
    if kind == "command":
        if not isinstance(step.get("args"), list) or not step["args"]:
            raise ValueError("command args must be a nonempty list")
        return None
    config = step["resolved"] if "resolved" in step else stage_config(step)
    if step.get("distributed", False) or step.get("shard", False):
        devices(step.get("devices", "auto"))
    if kind == "train" and "global_batch_size" in step:
        count = (
            len(devices(step.get("devices", "auto")))
            if step.get("distributed", False)
            else 1
        )
        training = config.setdefault("training", {})
        micro = _positive_int(
            training.get("per_device_train_batch_size", 8),
            "per_device_train_batch_size",
        )
        batch = _positive_int(step["global_batch_size"], "global_batch_size")
        if batch % (count * micro):
            raise ValueError(
                f"Global batch {batch} is not divisible by {count} GPUs x micro batch {micro}"
            )
        training["gradient_accumulation_steps"] = batch // (count * micro)
    return config


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
    config: dict[str, Any] | None,
    directory: str | Path,
    *,
    dry_run: bool = False,
) -> None:
    kind = step["kind"]
    if kind == "command":
        if not dry_run:
            for path in step.get("mkdir", []):
                Path(path).mkdir(parents=True, exist_ok=True)
        run_command(step["args"], cwd=step.get("cwd"), dry_run=dry_run)
        return

    config_path = Path(directory) / "config.yaml"
    OmegaConf.save(OmegaConf.create(config), config_path)
    if dry_run:
        print(OmegaConf.to_yaml(OmegaConf.create(config)), flush=True)
    if kind == "train":
        command = [sys.executable]
        env = os.environ.copy()
        if step.get("distributed", False):
            gpu_ids = devices(step.get("devices", "auto"))
            env["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
            command += [
                "-m",
                "torch.distributed.run",
                f"--nproc_per_node={len(gpu_ids)}",
                f"--master_port={step.get('master_port', 29501)}",
                "--module",
                "tinyllava.train.train",
            ]
        else:
            command += ["-m", "tinyllava.train.train"]
        run_command(command + ["--config", str(config_path)], env=env, dry_run=dry_run)
    elif not step.get("shard", False):
        run_command(
            [
                sys.executable,
                "-m",
                "tinyllava.eval.batch_generation",
                "--config",
                str(config_path),
            ],
            dry_run=dry_run,
        )
    else:
        gpu_ids = devices(step.get("devices", "auto"))
        commands, envs, outputs = [], [], []
        for index, gpu in enumerate(gpu_ids):
            shard = OmegaConf.create(config)
            output = str(Path(directory) / f"{index}.jsonl")
            shard.runtime.num_chunks = len(gpu_ids)
            shard.runtime.chunk_idx = index
            shard.runtime.device = "cuda:0"
            shard.output.answers_file = output
            path = Path(directory) / f"{index}.yaml"
            OmegaConf.save(shard, path)
            commands.append(
                [
                    sys.executable,
                    "-m",
                    "tinyllava.eval.batch_generation",
                    "--config",
                    str(path),
                ]
            )
            envs.append(dict(os.environ, CUDA_VISIBLE_DEVICES=gpu))
            outputs.append(output)
        if dry_run:
            for command in commands:
                run_command(command, dry_run=True)
            print(f"Merge {len(outputs)} shards -> {config['output']['answers_file']}")
        else:
            run_shards(commands, envs)
            merge_shards(outputs, config["output"]["answers_file"])
