"""Shared loading helpers for TinyLLaVA model components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from transformers import AutoConfig, AutoTokenizer

from tinyllava.data.processor.creation import create_tinyllava_processor
from tinyllava.model.configuration_tinyllava import TinyLlavaConfig
from tinyllava.model.llm import AutoLanguageModel
from tinyllava.model.modeling_tinyllava import (
    TinyLlavaForConditionalGeneration,
    build_connector,
)
from tinyllava.model.vision_tower import AutoVisionTowerModel
from tinyllava.model.vision_tower.registry import load_image_processor
from tinyllava.utils.arguments import ModelArguments
from tinyllava.utils.config import load_connector_config
from tinyllava.utils.logging import get_logger


logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class ComponentPaths:
    """Resolved component paths for TinyLLaVA loading."""

    language_model: str
    tokenizer: str
    vision_model: str
    image_processor: str


@dataclass(frozen=True, slots=True)
class ComponentPathSource:
    """Explain why a component resolved to a particular path."""

    component: str
    path: str | None
    source: str


@dataclass(frozen=True, slots=True)
class TinyLlavaModelBundle:
    """Loaded TinyLLaVA runtime objects shared by training and inference."""

    model: TinyLlavaForConditionalGeneration
    tokenizer: Any
    image_processor: Any
    processor: Any
    paths: ComponentPaths


def resolve_component_paths(model_args: ModelArguments) -> ComponentPaths:
    """Resolve the base components used to assemble a fresh TinyLLaVA model."""
    language_model_path = _required_path(
        model_args.model_name_or_path, "model_name_or_path"
    )
    vision_model_path = _required_path(
        model_args.vision_model_name_or_path, "vision_model_name_or_path"
    )
    tokenizer_path = model_args.tokenizer_name_or_path or language_model_path
    tokenizer_source = (
        "model_args.tokenizer_name_or_path"
        if model_args.tokenizer_name_or_path
        else "fallback to model_args.model_name_or_path"
    )

    paths = ComponentPaths(
        language_model=language_model_path,
        tokenizer=tokenizer_path,
        vision_model=vision_model_path,
        image_processor=vision_model_path,
    )
    _log_component_paths(
        [
            ComponentPathSource(
                "language_model",
                paths.language_model,
                "model_args.model_name_or_path",
            ),
            ComponentPathSource("tokenizer", paths.tokenizer, tokenizer_source),
            ComponentPathSource(
                "vision_model",
                paths.vision_model,
                "model_args.vision_model_name_or_path",
            ),
            ComponentPathSource(
                "image_processor",
                paths.image_processor,
                "fallback to model_args.vision_model_name_or_path",
            ),
            ComponentPathSource(
                "connector",
                None,
                "created from model_args.connector_config",
            ),
        ]
    )
    return paths


def load_model_config(
    model_args: ModelArguments,
    paths: ComponentPaths,
) -> TinyLlavaConfig:
    """Load TinyLLaVA composite config from resolved component paths."""
    text_config = AutoConfig.from_pretrained(
        paths.language_model,
        cache_dir=model_args.cache_dir,
    )
    vision_config = AutoConfig.from_pretrained(
        paths.vision_model,
        cache_dir=model_args.cache_dir,
    )
    return TinyLlavaConfig(
        text_config=text_config,
        vision_config=vision_config,
        connector_config=load_connector_config(model_args.connector_config),
        vision_feature_layer=model_args.vision_feature_layer,
        vision_feature_select_strategy=model_args.vision_feature_select_strategy,
    )


def load_tokenizer(
    model_args: ModelArguments,
    paths: ComponentPaths,
):
    """Load the tokenizer matching resolved language/checkpoint paths."""
    return AutoTokenizer.from_pretrained(
        paths.tokenizer,
        cache_dir=model_args.cache_dir,
        model_max_length=model_args.model_max_length,
        padding_side=model_args.tokenizer_padding_side,
        use_fast=model_args.tokenizer_use_fast,
    )


def load_model_components(
    model: TinyLlavaForConditionalGeneration,
    model_args: ModelArguments,
    paths: ComponentPaths,
    language_model_loading_kwargs: dict[str, Any] | None = None,
) -> None:
    """Load language, vision, and connector weights into a TinyLLaVA shell."""
    language_model_loading_kwargs = language_model_loading_kwargs or {}
    model.model.language_model = AutoLanguageModel.from_pretrained(
        paths.language_model,
        cache_dir=model_args.cache_dir,
        attn_implementation=model_args.attn_implementation,
        **language_model_loading_kwargs,
    )
    model.model.vision_tower = AutoVisionTowerModel.from_pretrained(
        paths.vision_model,
        cache_dir=model_args.cache_dir,
    )
    model.model.multi_modal_projector = build_connector(
        model.config
    )


def load_tinyllava_model_bundle(
    model_args: ModelArguments,
    *,
    language_model_loading_kwargs: dict[str, Any] | None = None,
    device: str | None = None,
) -> TinyLlavaModelBundle:
    """Assemble model, tokenizer, image processor, and processor from base components."""
    paths = resolve_component_paths(model_args)
    model = TinyLlavaForConditionalGeneration(load_model_config(model_args, paths))
    tokenizer = load_tokenizer(model_args, paths)
    load_model_components(
        model,
        model_args,
        paths,
        language_model_loading_kwargs=language_model_loading_kwargs,
    )
    model.tokenizer = tokenizer
    image_processor = load_image_processor(
        paths.image_processor,
        model_type=model.config.vision_config.model_type,
    )
    processor = create_tinyllava_processor(
        tokenizer=tokenizer,
        image_processor=image_processor,
        model=model,
    )
    model.processor = processor
    if device is not None:
        model = model.to(device)
    return TinyLlavaModelBundle(
        model=model,
        tokenizer=tokenizer,
        image_processor=image_processor,
        processor=processor,
        paths=paths,
    )


def load_tinyllava_checkpoint_bundle(
    model_path: str,
    *,
    device: str | None = None,
    **from_pretrained_kwargs: Any,
) -> TinyLlavaModelBundle:
    """Load a TinyLLaVA checkpoint saved by the standard Hugging Face flow."""
    logger.debug_rank0("Loading TinyLLaVA checkpoint from %s", model_path)
    model = TinyLlavaForConditionalGeneration.from_pretrained(
        model_path,
        **from_pretrained_kwargs,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    image_processor = load_image_processor(
        model_path,
        model_type=model.config.vision_config.model_type,
    )
    processor = create_tinyllava_processor(
        tokenizer=tokenizer,
        image_processor=image_processor,
        model=model,
    )
    model.tokenizer = tokenizer
    model.processor = processor
    if device is not None:
        model = model.to(device)
    paths = ComponentPaths(
        language_model=model_path,
        tokenizer=model_path,
        vision_model=model_path,
        image_processor=model_path,
    )
    _log_component_paths(
        [
            ComponentPathSource("model", model_path, "TinyLLaVA checkpoint"),
            ComponentPathSource("tokenizer", model_path, "checkpoint tokenizer"),
            ComponentPathSource(
                "image_processor",
                model_path,
                "checkpoint image processor",
            ),
        ]
    )
    return TinyLlavaModelBundle(
        model=model,
        tokenizer=tokenizer,
        image_processor=image_processor,
        processor=processor,
        paths=paths,
    )


def _log_component_paths(sources: list[ComponentPathSource]) -> None:
    logger.debug_rank0("Resolved TinyLLaVA component paths:")
    for source in sources:
        logger.debug_rank0(
            "  %s: %s (%s)",
            source.component,
            source.path if source.path is not None else "<none>",
            source.source,
        )


def _required_path(path: str | None, field_name: str) -> str:
    if path:
        return path
    raise ValueError(f"{field_name} must be set.")


__all__ = [
    "ComponentPathSource",
    "ComponentPaths",
    "TinyLlavaModelBundle",
    "load_model_components",
    "load_model_config",
    "load_tinyllava_checkpoint_bundle",
    "load_tinyllava_model_bundle",
    "load_tokenizer",
    "resolve_component_paths",
]
