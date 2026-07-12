import sys
from collections.abc import Sequence
from typing import Any

from omegaconf import OmegaConf

from tinyllava.utils.arguments import DataArguments, ModelArguments, TrainingArguments


TRAIN_CONFIG_SECTIONS = ("model", "data", "training", "peft")


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
        "overrides",
        nargs="*",
        help="OmegaConf dotlist overrides, for example training.output_dir=out.",
    )
    return parser.parse_args(args)


def _load_config_mapping(path: str) -> dict[str, Any]:
    config = OmegaConf.load(path)
    return _to_plain_container(config)


def _merge_overrides(config: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    merged = OmegaConf.merge(config, OmegaConf.from_dotlist(overrides))
    return _to_plain_container(merged)


def _to_plain_container(config) -> dict[str, Any]:
    plain = OmegaConf.to_container(config, resolve=True)
    if not isinstance(plain, dict):
        raise ValueError("Config must be a mapping.")
    return plain


def _require_mapping(section_name: str, section: Any) -> None:
    if not isinstance(section, dict):
        raise ValueError(f"Train config section '{section_name}' must be a mapping.")


__all__ = [
    "build_train_arguments",
    "load_connector_config",
    "parse_train_config",
]
