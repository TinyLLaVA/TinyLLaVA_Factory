"""Read and validate resolved evaluation arguments."""

import argparse
import sys
from collections.abc import Sequence
from typing import Any

from tinyllava.utils.config import load_config
from tinyllava.hparams import EVAL_SECTIONS
from tinyllava.hparams.arguments import (
    EvalArguments,
    EvalDataArguments,
    EvalGenerationArguments,
    EvalModelArguments,
    EvalOutputArguments,
    EvalRuntimeArguments,
)


def parse_eval_arguments(args: Sequence[str] | None = None) -> EvalArguments:
    """Read a resolved evaluation config and construct worker arguments.

    Dataset presets and output defaults are resolved by eval.workflow before
    launching this worker. This parser applies only field overrides.

    Args:
        args: Arguments including `--config PATH` and optional overrides.
            `None` reads `sys.argv[1:]`.

    Returns:
        Validated model, data, generation, runtime, and output settings.
    """

    cli_args = list(sys.argv[1:] if args is None else args)
    parsed_args = _parse_eval_args(cli_args)

    return build_eval_arguments(load_config(parsed_args.config, parsed_args.overrides))


def build_eval_arguments(config: dict[str, Any]) -> EvalArguments:
    """Construct evaluation arguments and validate batching and generation limits.

    Args:
        config: Mapping with `model`, `data`, `generation`, `runtime`, and `output`
            sections. Missing sections use their dataclass defaults.

    Returns:
        The grouped evaluation settings.

    Raises:
        ValueError: A section is unknown or not a mapping, a size is nonpositive,
            or the shard index is outside the configured range.
        TypeError: A section contains an unsupported argument name.
    """
    unknown_sections = sorted(set(config) - set(EVAL_SECTIONS))
    if unknown_sections:
        raise ValueError(f"Unknown eval config section(s): {unknown_sections}")

    sections = {name: config.get(name) or {} for name in EVAL_SECTIONS}
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


def _parse_eval_args(args: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate TinyLLaVA from a structured YAML config."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the resolved evaluation YAML written by the evaluation workflow.",
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
        raise ValueError(
            f"{config_kind} config section '{section_name}' must be a mapping."
        )


__all__ = [
    "build_eval_arguments",
    "parse_eval_arguments",
]
