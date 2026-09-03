import sys
from collections.abc import Sequence
from typing import Any

from omegaconf import OmegaConf

from tinyllava.utils.arguments import (
    DataArguments,
    EvalArguments,
    EvalDataArguments,
    EvalGenerationArguments,
    EvalModelArguments,
    EvalOutputArguments,
    EvalRuntimeArguments,
    ModelArguments,
    TrainingArguments,
)


TRAIN_CONFIG_SECTIONS = ("model", "data", "training", "peft")
EVAL_CONFIG_SECTIONS = ("model", "data", "generation", "runtime", "output")


def parse_train_config(
    args: Sequence[str] | None = None,
) -> tuple[ModelArguments, DataArguments, TrainingArguments]:
    """Parse TinyLLaVA training config.

    Usage:
        tinyllava/train/train.py --config configs/train/foo.yaml training.output_dir=...
    """

    cli_args = list(sys.argv[1:] if args is None else args)
    parsed_args = _parse_config_args(cli_args)

    config = _load_config_mapping(parsed_args.config)
    if parsed_args.overrides:
        config = _merge_overrides(config, parsed_args.overrides)
    config = _merge_launcher_arguments(
        config,
        deepspeed=parsed_args.deepspeed,
        local_rank=parsed_args.local_rank,
    )
    return build_train_arguments(config)


def build_train_arguments(
    config: dict[str, Any],
) -> tuple[ModelArguments, DataArguments, TrainingArguments]:
    unknown_sections = sorted(set(config) - set(TRAIN_CONFIG_SECTIONS))
    if unknown_sections:
        raise ValueError(f"Unknown train config section(s): {unknown_sections}")

    model_config = config.get("model") or {}
    data_config = config.get("data") or {}
    training_config = config.get("training") or {}
    peft_config = config.get("peft") or {}

    _require_mapping("model", model_config)
    _require_mapping("data", data_config)
    _require_mapping("training", training_config)
    _require_mapping("peft", peft_config)

    training_config = dict(training_config)
    if peft_config:
        if "peft_config" in training_config:
            raise ValueError("Use either top-level 'peft' or 'training.peft_config', not both.")
        training_config["peft_config"] = peft_config

    return (
        ModelArguments(**model_config),
        DataArguments(**data_config),
        TrainingArguments(**training_config),
    )


def parse_eval_config(args: Sequence[str] | None = None) -> EvalArguments:
    """Parse TinyLLaVA evaluation config."""

    cli_args = list(sys.argv[1:] if args is None else args)
    parsed_args = _parse_eval_config_args(cli_args)

    config = _load_config_mapping(parsed_args.config)
    if parsed_args.overrides:
        config = _merge_overrides(config, parsed_args.overrides)
    return build_eval_arguments(config)


def build_eval_arguments(config: dict[str, Any]) -> EvalArguments:
    unknown_sections = sorted(set(config) - set(EVAL_CONFIG_SECTIONS))
    if unknown_sections:
        raise ValueError(f"Unknown eval config section(s): {unknown_sections}")

    sections = {name: config.get(name) or {} for name in EVAL_CONFIG_SECTIONS}
    for name, section in sections.items():
        _require_mapping(name, section, config_kind="Eval")

    result = EvalArguments(
        model=EvalModelArguments(**sections["model"]),
        data=EvalDataArguments(**sections["data"]),
        generation=EvalGenerationArguments(**sections["generation"]),
        runtime=EvalRuntimeArguments(**sections["runtime"]),
        output=EvalOutputArguments(**sections["output"]),
    )
    _validate_eval_arguments(result)
    return result


def load_connector_config(path: str | None) -> dict[str, Any] | None:
    if path is None:
        return None

    connector_config = _load_config_mapping(path)
    if "model_type" not in connector_config:
        raise ValueError("connector_config must include a 'model_type' field.")
    return connector_config


def _parse_config_args(args: list[str]):
    import argparse

    parser = argparse.ArgumentParser(
        description="Train TinyLLaVA from a structured YAML config."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the root train YAML config.",
    )
    parser.add_argument(
        "--deepspeed",
        help=(
            "DeepSpeed config path. This is equivalent to the "
            "training.deepspeed YAML field."
        ),
    )
    parser.add_argument(
        "--local_rank",
        "--local-rank",
        type=int,
        default=None,
        help="Process-local rank injected by distributed launchers.",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="OmegaConf dotlist overrides, for example training.output_dir=out.",
    )
    return parser.parse_args(args)


def _parse_eval_config_args(args: list[str]):
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate TinyLLaVA from a structured YAML config."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the root eval YAML config.",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help=(
            "OmegaConf dotlist overrides, for example "
            "runtime.chunk_idx=1 model.model_name_or_path=output/model."
        ),
    )
    return parser.parse_args(args)


def _load_config_mapping(path: str) -> dict[str, Any]:
    config = OmegaConf.load(path)
    return _to_plain_container(config)


def _merge_overrides(config: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    merged = OmegaConf.merge(config, OmegaConf.from_dotlist(overrides))
    return _to_plain_container(merged)


def _merge_launcher_arguments(
    config: dict[str, Any],
    *,
    deepspeed: str | None,
    local_rank: int | None,
) -> dict[str, Any]:
    """Merge compatibility arguments supplied by distributed launchers."""
    if deepspeed is None and local_rank is None:
        return config

    training_config = config.get("training") or {}
    _require_mapping("training", training_config)
    training_config = dict(training_config)
    if deepspeed is not None:
        training_config["deepspeed"] = deepspeed
    if local_rank is not None:
        training_config["local_rank"] = local_rank

    config = dict(config)
    config["training"] = training_config
    return config


def _to_plain_container(config) -> dict[str, Any]:
    plain = OmegaConf.to_container(config, resolve=True)
    if not isinstance(plain, dict):
        raise ValueError("Config must be a mapping.")
    return plain


def _validate_eval_arguments(config: EvalArguments) -> None:
    if config.runtime.batch_size <= 0:
        raise ValueError("runtime.batch_size must be positive")
    if config.runtime.num_chunks <= 0:
        raise ValueError("runtime.num_chunks must be positive")
    if not 0 <= config.runtime.chunk_idx < config.runtime.num_chunks:
        raise ValueError("runtime.chunk_idx must be in [0, runtime.num_chunks)")
    if config.generation.max_new_tokens <= 0:
        raise ValueError("generation.max_new_tokens must be positive")
    if config.generation.num_beams <= 0:
        raise ValueError("generation.num_beams must be positive")


def _require_mapping(
    section_name: str,
    section: Any,
    *,
    config_kind: str = "Train",
) -> None:
    if not isinstance(section, dict):
        raise ValueError(f"{config_kind} config section '{section_name}' must be a mapping.")


__all__ = [
    "EVAL_CONFIG_SECTIONS",
    "build_train_arguments",
    "build_eval_arguments",
    "load_connector_config",
    "parse_eval_config",
    "parse_train_config",
]
