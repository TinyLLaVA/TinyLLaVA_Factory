"""Evaluation configuration, generation, and benchmark workflows."""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from tinyllava.utils.config import (
    compose_config,
    validate_name,
    validate_fields,
    resolve_interpolations,
)
from tinyllava import runner
from tinyllava.hparams import EVAL_SECTIONS

from tinyllava.eval.tasks.auto.workflow_auto import build_evaluation_workflow

_EVAL_METADATA = {
    "dataset_dir",
    "eval_dir",
    "output_dir",
    "benchmark",
    "devices",
    "shard",
}


def load_eval_config(
    path: str | Path, overrides: Sequence[str] = (), *, metadata: bool = False
) -> dict[str, Any]:
    """Load evaluation settings; metadata=True also returns workflow paths/options."""
    config = compose_config(path, overrides, data_kind="eval")
    validate_fields(config, EVAL_SECTIONS, _EVAL_METADATA, "eval")

    benchmark = validate_name(config.get("benchmark", Path(path).stem), "Benchmark")
    checkpoint = _set_evaluation_model_defaults(config)
    _set_evaluation_path_defaults(config, benchmark, checkpoint)
    config.setdefault("generation", {})
    config.setdefault("runtime", {})
    config = resolve_interpolations(config)
    config["benchmark"] = benchmark
    if metadata:
        return config
    return {section: config[section] for section in EVAL_SECTIONS if section in config}


def _set_evaluation_model_defaults(config: dict[str, Any]) -> str:
    model = config.setdefault("model", {})
    if not isinstance(model, dict):
        raise ValueError("Eval model must be a mapping with model_name_or_path")
    model.setdefault(
        "model_name_or_path",
        os.environ.get("MODEL_PATH", "output/qwen2_instruct/finetune"),
    )
    checkpoint = resolve_interpolations({"model": model})["model"]["model_name_or_path"]
    model.setdefault("model_id", os.environ.get("MODEL_NAME", Path(checkpoint).name))
    return checkpoint


def _set_evaluation_path_defaults(
    config: dict[str, Any], benchmark: str, checkpoint_path: str
) -> None:
    config.setdefault(
        "dataset_dir", os.environ.get("TINYLLAVA_DATASET_DIR", "datasets")
    )
    config.setdefault("eval_dir", os.environ.get("EVAL_DIR", "${dataset_dir}/eval"))
    model = config["model"]
    checkpoint = Path(checkpoint_path)
    # output/<experiment>/<stage> keeps its hierarchy; external checkpoints use model_id.
    try:
        model_directory = checkpoint.resolve().relative_to(Path("output").resolve())
    except ValueError:
        model_directory = Path(model.get("model_id") or checkpoint.name)
    config.setdefault(
        "output_dir", str(Path("output") / model_directory / "eval" / benchmark)
    )
    config.setdefault("output", {}).setdefault(
        "answers_file", "${output_dir}/answers.jsonl"
    )


def eval_plan(
    path: str | Path, overrides: Sequence[str] = ()
) -> list[tuple[str, dict[str, Any], dict[str, Any] | None]]:
    config = load_eval_config(path, overrides, metadata=True)
    if config.get("shard", False):
        runner.devices(config.get("devices", "auto"))
    benchmark = config["benchmark"]
    data = Path(config["eval_dir"]).expanduser().resolve()
    output = Path(config["output_dir"]).expanduser().resolve()
    answers = Path(config["output"]["answers_file"]).expanduser().resolve()
    config["output"]["answers_file"] = str(answers)
    generation = {k: v for k, v in config.items() if k in EVAL_SECTIONS}
    result = [
        (
            "generate",
            {
                "kind": "generate",
                "shard": config.get("shard", False),
                "devices": config.get("devices", "auto"),
            },
            generation,
        )
    ]
    result.extend(build_evaluation_workflow(benchmark, data, output, answers))
    return result


def run_generation_step(step, config, directory, *, dry_run=False):
    """Assign generation shards and merge their answers after all workers succeed."""
    if step["kind"] != "generate":
        return runner.run_step(step, dry_run=dry_run)
    config_path = Path(directory) / "config.yaml"
    OmegaConf.save(OmegaConf.create(config), config_path)
    if dry_run:
        print(OmegaConf.to_yaml(OmegaConf.create(config)), flush=True)
    if not step.get("shard", False):
        runner.run_command(
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
        gpu_ids = runner.devices(step.get("devices", "auto"))
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
                runner.run_command(command, dry_run=True)
            print(f"Merge {len(outputs)} shards -> {config['output']['answers_file']}")
        else:
            runner.run_shards(commands, envs)
            runner.merge_shards(outputs, config["output"]["answers_file"])


def run(
    path: str | Path,
    overrides: Sequence[str] = (),
    *,
    steps: Sequence[str] | None = None,
    dry_run: bool = False,
) -> None:
    """Build generation and benchmark stages, then execute the requested selection."""
    runner.run_plan(
        eval_plan(path, overrides),
        steps=steps,
        dry_run=dry_run,
        execute_step=run_generation_step,
    )
