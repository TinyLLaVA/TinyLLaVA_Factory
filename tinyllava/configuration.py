"""Compose model/data presets without importing training dependencies."""

from __future__ import annotations

import os
from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

CONFIG_ROOT = Path(__file__).resolve().parents[1] / "configs"
TRAIN_SECTIONS = ("model", "data", "training", "peft")
EVAL_SECTIONS = ("model", "data", "generation", "runtime", "output")


def read_mapping(path: str | Path) -> dict[str, Any]:
    config = OmegaConf.load(path)
    if not OmegaConf.is_dict(config):
        raise ValueError(f"Config must be a mapping: {path}")
    return OmegaConf.to_container(config, resolve=False)


def merge(*configs: dict[str, Any] | DictConfig) -> dict[str, Any]:
    return OmegaConf.to_container(OmegaConf.merge(*configs), resolve=False)


def resolve(config: dict[str, Any]) -> dict[str, Any]:
    return OmegaConf.to_container(OmegaConf.create(config), resolve=True)


def read_spec(path: str | Path, overrides: Sequence[str] = ()) -> dict[str, Any]:
    config = read_mapping(path)

    def normalize_model(mapping: dict[str, Any]) -> None:
        if isinstance(mapping.get("model"), str):
            mapping["model"] = {"name": mapping["model"]}
        for stage in mapping.get("stages", {}).values():
            if isinstance(stage, dict):
                normalize_model(stage)

    normalize_model(config)
    for override in overrides:
        config = merge(config, OmegaConf.from_dotlist([override]))
        normalize_model(config)
    return config


def safe_name(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or Path(value).name != value
        or value in (".", "..")
    ):
        raise ValueError(f"{label} must be a nonempty name without path separators")
    return value


def preset(group: str, name: str) -> dict[str, Any]:
    return read_mapping(CONFIG_ROOT / group / f"{safe_name(name, group)}.yaml")


def compose_train(
    spec: dict[str, Any], *, name: str, stage: str, previous: str | None = None
) -> dict[str, Any]:
    spec = deepcopy(spec)
    metadata = {
        "recipe",
        "dataset",
        "dataset_dir",
        "name",
        "stage",
        "output_root",
        "checkpoint",
        "from",
        "launcher",
    }
    unknown = set(spec) - set(TRAIN_SECTIONS) - metadata
    if unknown:
        raise ValueError(f"Unknown train config fields: {sorted(unknown)}")
    model = spec.get("model", {})
    if isinstance(model, str):
        model = {"name": model}
    model = dict(model)
    model_name = model.pop("name", None)
    base_model = preset("models", model_name) if model_name else {}
    recipe = preset("recipes", spec["recipe"]) if spec.get("recipe") else {}
    dataset = preset("data", spec["dataset"]) if spec.get("dataset") else {}
    config = merge(
        {"model": base_model, "data": dataset},
        recipe,
        {k: v for k, v in spec.items() if k in TRAIN_SECTIONS and k != "model"},
        {"model": model},
    )
    name = safe_name(spec.get("name", name), "Experiment name")
    stage = safe_name(spec.get("stage", stage), "Stage name")
    root = spec.get("output_root", "output")
    config["dataset_dir"] = spec.get(
        "dataset_dir", os.environ.get("TINYLLAVA_DATASET_DIR", "datasets")
    )
    training = config.setdefault("training", {})
    training.setdefault("output_dir", str(Path(root) / name / stage))
    training.setdefault("run_name", f"{name}/{stage}")
    checkpoint = spec.get(
        "checkpoint",
        config.get("model", {}).get("pretrained_model_name_or_path") or previous,
    )
    if "checkpoint" in spec and spec["checkpoint"] is None:
        config.get("model", {}).pop("pretrained_model_name_or_path", None)
    if "from" in spec and previous is None and "checkpoint" not in spec:
        checkpoint = str(Path(root) / name / safe_name(spec["from"], "Source stage"))
    if checkpoint is not None:
        config.setdefault("model", {})["pretrained_model_name_or_path"] = checkpoint
    config = resolve(config)
    config.pop("dataset_dir")
    if config.get("model", {}).get("pretrained_model_name_or_path"):
        for field in (
            "language_model_name_or_path",
            "vision_model_name_or_path",
            "connector_config",
        ):
            config["model"].pop(field, None)
    return config


def load_train_config(
    path: str | Path, overrides: Sequence[str] = ()
) -> dict[str, Any]:
    spec = read_spec(path, overrides)
    model = spec.get("model", {})
    model_name = model if isinstance(model, str) else model.get("name", Path(path).stem)
    return compose_train(spec, name=spec.get("name", model_name), stage=Path(path).stem)


def training_stages(
    path: str | Path, overrides: Sequence[str] = ()
) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
    spec = read_spec(path, overrides)
    if "stages" not in spec:
        return [
            (
                spec.get("stage", Path(path).stem),
                load_train_config(path, overrides),
                spec.get("launcher", {}),
            )
        ]
    stages = spec.pop("stages")
    if not isinstance(stages, dict) or not stages:
        raise ValueError("stages must be a nonempty mapping")
    name = spec.get("name", Path(path).stem)
    outputs, result = {}, []
    output_paths = set()
    previous = None
    for stage, values in stages.items():
        safe_name(stage, "Stage name")
        if not isinstance(values, dict):
            raise ValueError(f"Stage {stage} must be a mapping")
        current = merge(spec, values)
        source = previous
        if "from" in current:
            reference = current["from"]
            if reference not in outputs:
                raise ValueError(
                    f"Stage {stage} refers to an unknown or later stage: {reference}"
                )
            source = outputs[reference]
        config = compose_train(current, name=name, stage=stage, previous=source)
        output = config["training"]["output_dir"]
        canonical_output = Path(output).expanduser().resolve()
        if canonical_output in output_paths:
            raise ValueError(f"Stages must have distinct output directories: {output}")
        output_paths.add(canonical_output)
        outputs[stage] = output
        previous = output
        result.append((stage, config, current.get("launcher", {})))
    return result


def load_eval_config(
    path: str | Path, overrides: Sequence[str] = (), *, metadata: bool = False
) -> dict[str, Any]:
    spec = read_spec(path, overrides)
    dataset_name = spec.pop("dataset", None)
    data_config = preset("data/eval", dataset_name) if dataset_name else {}
    config = merge(data_config, spec)
    config.setdefault(
        "dataset_dir", os.environ.get("TINYLLAVA_DATASET_DIR", "datasets")
    )
    config.setdefault("eval_dir", os.environ.get("EVAL_DIR", "${dataset_dir}/eval"))
    model = config.setdefault("model", {})
    if "name" in model:
        model["model_name_or_path"] = model.pop("name")
    model.setdefault(
        "model_name_or_path",
        os.environ.get("MODEL_PATH", "output/qwen2_instruct/finetune"),
    )
    checkpoint = resolve({"model": model})["model"]["model_name_or_path"]
    model.setdefault("model_id", os.environ.get("MODEL_NAME", Path(checkpoint).name))
    benchmark = safe_name(
        config.get("benchmark", dataset_name or Path(path).stem), "Benchmark"
    )
    try:
        identity = Path(checkpoint).resolve().relative_to(Path("output").resolve())
    except ValueError:
        identity = Path(model.get("model_id") or Path(checkpoint).name)
    config.setdefault("output_dir", str(Path("output") / identity / "eval" / benchmark))
    config.setdefault("output", {}).setdefault(
        "answers_file", "${output_dir}/answers.jsonl"
    )
    config.setdefault("generation", {})
    config.setdefault("runtime", {})
    config = resolve(config)
    config["benchmark"] = benchmark
    allowed_metadata = {
        "dataset_dir",
        "eval_dir",
        "output_dir",
        "benchmark",
        "devices",
        "shard",
    }
    unknown = set(config) - set(EVAL_SECTIONS) - allowed_metadata
    if unknown:
        raise ValueError(f"Unknown eval config fields: {sorted(unknown)}")
    if metadata:
        return config
    return {k: v for k, v in config.items() if k in EVAL_SECTIONS}
