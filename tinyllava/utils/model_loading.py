"""Shared loading helpers for TinyLLaVA model components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from transformers import AutoConfig, AutoTokenizer

from tinyllava.data.chat_template.loading import resolve_chat_template
from tinyllava.data.processor.creation import create_tinyllava_processor
from tinyllava.model.configuration_tinyllava import TinyLlavaConfig
from tinyllava.model.modeling_tinyllava import TinyLlavaForConditionalGeneration
from tinyllava.model.vision_tower.registry import load_image_processor
from tinyllava.utils.arguments import ModelArguments
from tinyllava.utils.config import load_connector_config
from tinyllava.utils.logging import get_logger


logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class ComponentPaths:
    """Resolved Hugging Face repository IDs or local component paths.

    Attributes:
        pretrained_model: Composite checkpoint, or `None` when assembling components.
        language_model: Language-model source, or the composite checkpoint.
        tokenizer: Tokenizer source, including any explicit override.
        vision_model: Vision-tower source, or the composite checkpoint.
        image_processor: Image-preprocessing assets matching the vision tower.
    """

    pretrained_model: str | None
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
    """Model and preprocessing objects loaded from matching component sources.

    Attributes:
        model: Composite TinyLLaVA model.
        tokenizer: Tokenizer shared with the multimodal processor.
        image_processor: Image preprocessor shared with the multimodal processor.
        processor: Combined processor used for training and generation.
        paths: Resolved sources for the loaded components.
    """

    model: TinyLlavaForConditionalGeneration
    tokenizer: Any
    image_processor: Any
    processor: Any
    paths: ComponentPaths


def resolve_component_paths(model_args: ModelArguments) -> ComponentPaths:
    """Resolve model and preprocessing sources without loading their weights.

    Args:
        model_args: A composite checkpoint path or separate base-component paths.
            An explicit tokenizer path overrides either mode's default.

    Returns:
        Paths for the model, tokenizer, vision tower, and image processor.

    Raises:
        ValueError: Component assembly is requested without a language or vision path.
    """
    if model_args.pretrained_model_name_or_path:
        checkpoint_path = model_args.pretrained_model_name_or_path
        tokenizer_path = model_args.tokenizer_name_or_path or checkpoint_path
        paths = ComponentPaths(
            pretrained_model=checkpoint_path,
            language_model=checkpoint_path,
            tokenizer=tokenizer_path,
            vision_model=checkpoint_path,
            image_processor=checkpoint_path,
        )
        _log_component_paths(
            [
                ComponentPathSource(
                    "model",
                    checkpoint_path,
                    "model_args.pretrained_model_name_or_path",
                ),
                ComponentPathSource(
                    "tokenizer",
                    tokenizer_path,
                    (
                        "model_args.tokenizer_name_or_path"
                        if model_args.tokenizer_name_or_path
                        else "TinyLLaVA checkpoint"
                    ),
                ),
                ComponentPathSource(
                    "image_processor",
                    checkpoint_path,
                    "TinyLLaVA checkpoint",
                ),
            ]
        )
        return paths

    language_model_path = _required_path(
        model_args.language_model_name_or_path,
        "language_model_name_or_path",
    )
    vision_model_path = _required_path(
        model_args.vision_model_name_or_path, "vision_model_name_or_path"
    )
    tokenizer_path = model_args.tokenizer_name_or_path or language_model_path
    tokenizer_source = (
        "model_args.tokenizer_name_or_path"
        if model_args.tokenizer_name_or_path
        else "fallback to model_args.language_model_name_or_path"
    )

    paths = ComponentPaths(
        pretrained_model=None,
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
                "model_args.language_model_name_or_path",
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
    if paths.pretrained_model is not None:
        return TinyLlavaConfig.from_pretrained(
            paths.pretrained_model,
            cache_dir=model_args.cache_dir,
        )

    text_config = AutoConfig.from_pretrained(
        paths.language_model,
        cache_dir=model_args.cache_dir,
    )
    vision_config = AutoConfig.from_pretrained(
        paths.vision_model,
        cache_dir=model_args.cache_dir,
    )
    vision_config = getattr(vision_config, "vision_config", vision_config)
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


def load_training_model(
    model_args: ModelArguments,
    paths: ComponentPaths,
    model_config: TinyLlavaConfig,
    *,
    language_model_loading_kwargs: dict[str, Any] | None = None,
) -> TinyLlavaForConditionalGeneration:
    """Load composite weights or assemble pretrained language and vision components.

    Args:
        model_args: Model cache and attention-backend settings.
        paths: Component sources returned by `resolve_component_paths`.
        model_config: Composite configuration matching those sources.
        language_model_loading_kwargs: Loading overrides such as dtype or quantization.
            Applied to the composite loader when loading a complete checkpoint.

    Returns:
        A model with loaded weights. Component assembly initializes a new connector.
    """
    loading_kwargs = dict(language_model_loading_kwargs or {})
    loading_kwargs.setdefault("cache_dir", model_args.cache_dir)
    loading_kwargs.setdefault(
        "attn_implementation", model_args.attn_implementation
    )

    if paths.pretrained_model is not None:
        # Transformers recursively applies a string implementation to every
        # sub-config in a composite checkpoint. Scope it to the causal LM so
        # connector models without attention are left untouched.
        attn_implementation = loading_kwargs.get("attn_implementation")
        if isinstance(attn_implementation, str):
            loading_kwargs["attn_implementation"] = {
                "text_config": attn_implementation
            }
        logger.info_rank0(
            "Loading complete TinyLLaVA checkpoint from %s.",
            paths.pretrained_model,
        )
        return TinyLlavaForConditionalGeneration.from_pretrained(
            paths.pretrained_model,
            config=model_config,
            **loading_kwargs,
        )

    return TinyLlavaForConditionalGeneration.from_pretrained_components(
        model_config,
        language_model_name_or_path=paths.language_model,
        vision_model_name_or_path=paths.vision_model,
        language_model_loading_kwargs=loading_kwargs,
        vision_model_loading_kwargs={"cache_dir": model_args.cache_dir},
    )


def load_tinyllava_model_bundle(
    model_args: ModelArguments,
    *,
    language_model_loading_kwargs: dict[str, Any] | None = None,
    device: str | None = None,
) -> TinyLlavaModelBundle:
    """Load a model and its processors from a checkpoint or separate components.

    Args:
        model_args: Component paths, tokenizer settings, and optional chat template.
        language_model_loading_kwargs: Model-loading options such as dtype or quantization.
        device: Optional device for the loaded model; `None` keeps the loader placement.

    Returns:
        A bundle with matching model and preprocessing objects. The tokenizer and
        processor are also attached to the model.
    """
    paths = resolve_component_paths(model_args)
    model = load_training_model(
        model_args,
        paths,
        load_model_config(model_args, paths),
        language_model_loading_kwargs=language_model_loading_kwargs,
    )
    tokenizer = load_tokenizer(model_args, paths)
    model.tokenizer = tokenizer
    image_processor = load_image_processor(
        paths.image_processor,
        model_type=model.config.vision_config.model_type,
    )
    processor = create_tinyllava_processor(
        tokenizer=tokenizer,
        image_processor=image_processor,
        model=model,
        chat_template=resolve_chat_template(
            chat_template=model_args.chat_template,
            chat_template_path=model_args.chat_template_path,
        ),
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
    """Load a composite checkpoint together with its saved preprocessing assets.

    Args:
        model_path: Local directory or Hugging Face repository containing model,
            tokenizer, and image-processor files.
        device: Optional target device; `None` keeps the loader's placement.
        **from_pretrained_kwargs: Options forwarded to the model's `from_pretrained`
            call, such as dtype. Tokenizer and image-processor loading use `model_path`.

    Returns:
        Matching model, tokenizer, image processor, combined processor, and source paths.
    """
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
        pretrained_model=model_path,
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
    "load_model_config",
    "load_training_model",
    "load_tinyllava_checkpoint_bundle",
    "load_tinyllava_model_bundle",
    "load_tokenizer",
    "resolve_component_paths",
]
