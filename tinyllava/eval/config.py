"""Structured YAML configuration for benchmark generation."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from omegaconf import OmegaConf


EVAL_CONFIG_SECTIONS = ("model", "data", "generation", "runtime", "output")


@dataclass
class EvalModelArguments:
    model_path: str = "output/tinyllava"
    model_id: str | None = None


@dataclass
class EvalDataArguments:
    adapter: str = "vqa"
    question_file: str = "tables/question.jsonl"
    image_folder: str = ""
    single_pred_prompt: bool = False


@dataclass
class EvalGenerationArguments:
    temperature: float = 0.2
    top_p: float | None = None
    num_beams: int = 1
    max_new_tokens: int = 128


@dataclass
class EvalRuntimeArguments:
    device: str | None = None
    num_chunks: int = 1
    chunk_idx: int = 0


@dataclass
class EvalOutputArguments:
    answers_file: str = "answer.jsonl"


@dataclass
class EvalConfig:
    model: EvalModelArguments
    data: EvalDataArguments
    generation: EvalGenerationArguments
    runtime: EvalRuntimeArguments
    output: EvalOutputArguments


def build_eval_config(config: dict[str, Any]) -> EvalConfig:
    unknown_sections = sorted(set(config) - set(EVAL_CONFIG_SECTIONS))
    if unknown_sections:
        raise ValueError(f"Unknown eval config section(s): {unknown_sections}")

    sections = {name: config.get(name) or {} for name in EVAL_CONFIG_SECTIONS}
    for name, section in sections.items():
        if not isinstance(section, dict):
            raise ValueError(f"Eval config section {name!r} must be a mapping.")

    result = EvalConfig(
        model=EvalModelArguments(**sections["model"]),
        data=EvalDataArguments(**sections["data"]),
        generation=EvalGenerationArguments(**sections["generation"]),
        runtime=EvalRuntimeArguments(**sections["runtime"]),
        output=EvalOutputArguments(**sections["output"]),
    )
    _validate_eval_config(result)
    return result


def parse_eval_config(args: Sequence[str] | None = None) -> EvalConfig:
    cli_args = list(sys.argv[1:] if args is None else args)
    parser = argparse.ArgumentParser(
        description="Evaluate TinyLLaVA from a structured YAML config."
    )
    parser.add_argument("--config", required=True, help="Path to an eval YAML config.")
    parser.add_argument(
        "overrides",
        nargs="*",
        help=(
            "OmegaConf dotlist overrides, for example "
            "runtime.chunk_idx=1 model.model_path=output/model."
        ),
    )
    parsed = parser.parse_args(cli_args)

    config = OmegaConf.load(parsed.config)
    if parsed.overrides:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(parsed.overrides))
    plain = OmegaConf.to_container(config, resolve=True)
    if not isinstance(plain, dict):
        raise ValueError("Eval config must be a mapping.")
    return build_eval_config(plain)


def _validate_eval_config(config: EvalConfig) -> None:
    if config.runtime.num_chunks <= 0:
        raise ValueError("runtime.num_chunks must be positive")
    if not 0 <= config.runtime.chunk_idx < config.runtime.num_chunks:
        raise ValueError("runtime.chunk_idx must be in [0, runtime.num_chunks)")
    if config.generation.max_new_tokens <= 0:
        raise ValueError("generation.max_new_tokens must be positive")
    if config.generation.num_beams <= 0:
        raise ValueError("generation.num_beams must be positive")


__all__ = [
    "EVAL_CONFIG_SECTIONS",
    "EvalConfig",
    "EvalDataArguments",
    "EvalGenerationArguments",
    "EvalModelArguments",
    "EvalOutputArguments",
    "EvalRuntimeArguments",
    "build_eval_config",
    "parse_eval_config",
]
