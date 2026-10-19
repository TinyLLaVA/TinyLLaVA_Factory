"""Training configuration, stage dependencies, and distributed execution."""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from tinyllava import runner
from tinyllava.train.algorithms import get_algorithm
from tinyllava.utils.config import (
    compose_config,
    merge_mappings,
    resolve_interpolations,
    validate_fields,
    validate_name,
)

_TRAIN_METADATA = {
    "algorithm",
    "dataset_dir",
    "experiment_name",
    "stage",
    "output_root",
    "checkpoint",
    "from_stage",
    "launcher",
}


def load_train_config(
    path: str | Path, overrides: Sequence[str] = ()
) -> dict[str, Any]:
    """Load one training stage, including presets and ordered CLI overrides."""
    experiment_config = compose_config(path, overrides)
    return _compose_single_training_stage(experiment_config, Path(path).stem)


def training_stages(
    path: str | Path, overrides: Sequence[str] = ()
) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
    """Return (stage name, resolved config, launcher options) in YAML order.

    Algorithm and data are stage-local. Shared model, training, and launcher
    defaults are merged before stage fields. Each stage inherits the
    preceding output checkpoint unless it selects an earlier stage with 'from_stage'
    or specifies a checkpoint. 'checkpoint: null' starts a fresh model.
    """
    experiment_config = compose_config(path, overrides)
    if "stages" not in experiment_config:
        stage_name = experiment_config.get("stage", Path(path).stem)
        config = _compose_single_training_stage(experiment_config, Path(path).stem)
        return [(stage_name, config, experiment_config.get("launcher", {}))]

    stages = experiment_config.pop("stages")
    if not isinstance(stages, dict) or not stages:
        raise ValueError("stages must be a nonempty mapping")

    validate_fields(
        experiment_config,
        ("model", "training", "launcher"),
        {"dataset_dir", "experiment_name", "output_root"},
        "shared experiment",
    )
    experiment_name = experiment_config.get("experiment_name", Path(path).stem)
    completed_outputs: dict[str, str] = {}
    output_directories: set[Path] = set()
    previous_checkpoint = None
    result = []
    for stage_name, stage_overrides in stages.items():
        validate_name(stage_name, "Stage name")
        if not isinstance(stage_overrides, dict):
            raise ValueError(f"Stage {stage_name} must be a mapping")
        stage_config = merge_mappings(experiment_config, stage_overrides)

        source_checkpoint = previous_checkpoint
        if "from_stage" in stage_config:
            source_stage = stage_config["from_stage"]
            if source_stage not in completed_outputs:
                raise ValueError(
                    f"Stage {stage_name} refers to an unknown or later stage: {source_stage}"
                )
            source_checkpoint = completed_outputs[source_stage]

        config = _compose_training_stage(
            stage_config,
            experiment_name=experiment_name,
            stage_name=stage_name,
            source_checkpoint=source_checkpoint,
        )
        output_dir = config["training"]["output_dir"]
        canonical_directory = Path(output_dir).expanduser().resolve()
        if canonical_directory in output_directories:
            raise ValueError(
                f"Stages must have distinct output directories: {output_dir}"
            )
        output_directories.add(canonical_directory)
        completed_outputs[stage_name] = output_dir
        previous_checkpoint = output_dir
        result.append((stage_name, config, stage_config.get("launcher", {})))
    return result


def _compose_single_training_stage(
    experiment_config: dict[str, Any], filename_stem: str
) -> dict[str, Any]:
    return _compose_training_stage(
        experiment_config, experiment_name=filename_stem, stage_name=filename_stem
    )


def _compose_training_stage(
    experiment_config: dict[str, Any],
    *,
    experiment_name: str,
    stage_name: str,
    source_checkpoint: str | None = None,
) -> dict[str, Any]:
    experiment_name = validate_name(
        experiment_config.get("experiment_name", experiment_name), "Experiment name"
    )
    stage_name = validate_name(experiment_config.get("stage", stage_name), "Stage name")
    experiment_dir = (
        Path(experiment_config.get("output_root", "output")) / experiment_name
    )

    algorithm = get_algorithm(experiment_config.get("algorithm", "sft"))
    validate_fields(experiment_config, algorithm.SECTIONS, _TRAIN_METADATA, "train")
    config = {
        key: value
        for key, value in experiment_config.items()
        if key in algorithm.SECTIONS
    }
    config["algorithm"] = experiment_config.get("algorithm", "sft")
    config.setdefault("model", {})
    config["dataset_dir"] = experiment_config.get(
        "dataset_dir", os.environ.get("TINYLLAVA_DATASET_DIR", "datasets")
    )
    training = config.setdefault("training", {})
    training.setdefault("output_dir", str(experiment_dir / stage_name))
    training.setdefault("run_name", f"{experiment_name}/{stage_name}")
    _set_training_checkpoint(
        config, experiment_config, source_checkpoint, experiment_dir
    )

    # A composite checkpoint supplies these components itself.
    if config["model"].get("pretrained_model_name_or_path"):
        for field in (
            "language_model_name_or_path",
            "vision_model_name_or_path",
            "connector",
        ):
            config["model"].pop(field, None)
    config = resolve_interpolations(config)
    config.pop("dataset_dir")
    algorithm.validate_config(config)
    return config


def _set_training_checkpoint(
    config: dict[str, Any],
    experiment_config: dict[str, Any],
    source_checkpoint: str | None,
    experiment_dir: Path,
) -> None:
    """Select explicit checkpoint, standalone 'from_stage', model checkpoint, or prior output."""
    model = config["model"]
    if "checkpoint" in experiment_config:
        checkpoint = experiment_config[
            "checkpoint"
        ]  # None explicitly requests a fresh model.
    elif "from_stage" in experiment_config and source_checkpoint is None:
        source_stage = validate_name(experiment_config["from_stage"], "Source stage")
        checkpoint = str(experiment_dir / source_stage)
    else:
        checkpoint = model.get("pretrained_model_name_or_path") or source_checkpoint

    if "checkpoint" in experiment_config and checkpoint is None:
        model.pop("pretrained_model_name_or_path", None)
    elif checkpoint is not None:
        model["pretrained_model_name_or_path"] = checkpoint


def train_plan(
    path: str | Path, overrides: Sequence[str] = ()
) -> list[tuple[str, dict[str, Any], dict[str, Any] | None]]:
    result = []
    for name, config, launcher in training_stages(path, overrides):
        allowed = {"devices", "distributed", "master_port"}
        unknown = set(launcher) - allowed
        if unknown:
            raise ValueError(f"Unknown launcher fields: {sorted(unknown)}")
        world_size = (
            len(runner.devices(launcher.get("devices", "auto")))
            if launcher.get("distributed", False)
            else 1
        )
        config = get_algorithm(config["algorithm"]).prepare_config(
            config, world_size=world_size
        )
        step = {"kind": "train", **launcher}
        result.append((name, step, config))
    return result


def _run_training_step(step, config, directory, *, dry_run=False):
    config_path = Path(directory) / "config.yaml"
    OmegaConf.save(OmegaConf.create(config), config_path)
    if dry_run:
        print(OmegaConf.to_yaml(OmegaConf.create(config)), flush=True)
    command = [sys.executable]
    env = os.environ.copy()
    if step.get("distributed", False):
        gpu_ids = runner.devices(step.get("devices", "auto"))
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
    runner.run_command(
        command + ["--config", str(config_path)], env=env, dry_run=dry_run
    )


def run(
    path: str | Path,
    overrides: Sequence[str] = (),
    *,
    steps: Sequence[str] | None = None,
    dry_run: bool = False,
) -> None:
    """Build all training stages before executing the requested selection."""
    runner.run_plan(
        train_plan(path, overrides),
        steps=steps,
        dry_run=dry_run,
        execute_step=_run_training_step,
    )
