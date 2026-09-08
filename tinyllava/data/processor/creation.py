from __future__ import annotations

from typing import Any

from transformers.models.auto.processing_auto import PROCESSOR_MAPPING

from tinyllava.data.processor.auto import AutoProcessor  # noqa: F401 - registers TinyLLaVA with HF AutoProcessor.
from tinyllava.data.chat_template import inject_tinyllava_anchors
from tinyllava.model.configuration_tinyllava import TinyLlavaConfig
from tinyllava.utils.constants import DEFAULT_IMAGE_TOKEN


def create_tinyllava_processor(
    *,
    tokenizer: Any,
    image_processor: Any,
    model: Any | None = None,
    chat_template: str | None = None,
) -> Any:
    """Build the registered multimodal processor and synchronize its image token.

    The tokenizer is updated in place with the image token and template anchors.
    When a model is supplied, its image-token ID and embedding size are synchronized.

    Args:
        tokenizer: Tokenizer used to encode conversations.
        image_processor: Image preprocessor matching the vision tower.
        model: Model providing vision settings and processor registration. Without
            a model, use the default TinyLLaVA configuration.
        chat_template: Jinja template override; otherwise use the tokenizer template.

    Returns:
        A registered processor combining the tokenizer and image processor.

    Raises:
        ValueError: The image token cannot be encoded as one token, or the vision
            feature selection conflicts with the backbone's additional tokens.
    """
    ensure_image_token(tokenizer, model=model)

    template = chat_template or getattr(tokenizer, "chat_template", None)
    if template is not None:
        template = inject_tinyllava_anchors(template).chat_template
        tokenizer.chat_template = template

    config = getattr(model, "config", None)
    vision_config = getattr(config, "vision_config", None)
    vision_feature_select_strategy = _get_attr(
        config, "vision_feature_select_strategy", default="default"
    )
    num_additional_image_tokens = _get_num_additional_image_tokens(vision_config)
    _validate_image_feature_configuration(
        num_additional_image_tokens=num_additional_image_tokens,
        vision_feature_select_strategy=vision_feature_select_strategy,
    )

    auto_config = config if config is not None and type(config) in PROCESSOR_MAPPING else TinyLlavaConfig()
    processor_cls = PROCESSOR_MAPPING[type(auto_config)]
    return processor_cls(
        image_processor=image_processor,
        tokenizer=tokenizer,
        patch_size=_get_attr(vision_config, "patch_size", default=14),
        vision_feature_select_strategy=vision_feature_select_strategy,
        num_additional_image_tokens=num_additional_image_tokens,
        chat_template=template,
        image_token=DEFAULT_IMAGE_TOKEN,
    )


def ensure_image_token(tokenizer: Any, model: Any | None = None) -> int:
    image_token_id = _encode_single_token(tokenizer, DEFAULT_IMAGE_TOKEN)
    if image_token_id is None and hasattr(tokenizer, "add_special_tokens"):
        tokenizer.add_special_tokens({"additional_special_tokens": [DEFAULT_IMAGE_TOKEN]})
        image_token_id = _encode_single_token(tokenizer, DEFAULT_IMAGE_TOKEN)

    if image_token_id is None:
        raise ValueError(f"Tokenizer cannot encode TinyLLaVA image token {DEFAULT_IMAGE_TOKEN!r}.")

    tokenizer.image_token = DEFAULT_IMAGE_TOKEN
    if model is not None and hasattr(model, "resize_token_embeddings"):
        try:
            model.resize_token_embeddings(len(tokenizer))
        except TypeError:
            pass
    config = getattr(model, "config", None)
    if config is not None:
        config.image_token_id = image_token_id
    return image_token_id


def _encode_single_token(tokenizer: Any, token: str) -> int | None:
    try:
        token_ids = tokenizer.encode(token, add_special_tokens=False)
    except TypeError:
        token_ids = tokenizer.encode(token)
    except Exception:
        return None

    if isinstance(token_ids, int):
        return token_ids
    if len(token_ids) == 1:
        return int(token_ids[0])
    return None


def _get_attr(obj: Any, name: str, default: Any) -> Any:
    if obj is None:
        return default
    return getattr(obj, name, default)


def _get_num_additional_image_tokens(vision_config: Any) -> int:
    value = _get_attr(vision_config, "num_additional_image_tokens", default=None)
    if value is not None:
        return value

    model_type = _get_attr(vision_config, "model_type", default="")
    if "siglip" in model_type:
        return 0
    return 1


def _validate_image_feature_configuration(
    *,
    num_additional_image_tokens: int,
    vision_feature_select_strategy: str,
) -> None:
    if vision_feature_select_strategy not in {"default", "full"}:
        raise ValueError(
            "vision_feature_select_strategy must be either 'default' or 'full', "
            f"got {vision_feature_select_strategy!r}."
        )
    if (
        isinstance(num_additional_image_tokens, bool)
        or not isinstance(num_additional_image_tokens, int)
        or num_additional_image_tokens < 0
    ):
        raise ValueError(
            "num_additional_image_tokens must be a non-negative integer, "
            f"got {num_additional_image_tokens!r}."
        )
    if vision_feature_select_strategy == "default" and num_additional_image_tokens == 0:
        raise ValueError(
            "Invalid image feature configuration: vision_feature_select_strategy='default' "
            "drops the first vision feature, but num_additional_image_tokens=0 means the "
            "vision tower has no CLS/additional token to drop. Use "
            "vision_feature_select_strategy='full' for patch-only backbones such as SigLIP."
        )


__all__ = ["create_tinyllava_processor", "ensure_image_token"]
