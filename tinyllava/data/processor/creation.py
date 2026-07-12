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
    """Create the HF Auto-selected processor used by TinyLLaVA."""
    ensure_image_token(tokenizer, model=model)

    template = chat_template or getattr(tokenizer, "chat_template", None)
    if template is not None:
        template = inject_tinyllava_anchors(template).chat_template
        tokenizer.chat_template = template

    config = getattr(model, "config", None)
    vision_config = getattr(config, "vision_config", None)

    auto_config = config if config is not None and type(config) in PROCESSOR_MAPPING else TinyLlavaConfig()
    processor_cls = PROCESSOR_MAPPING[type(auto_config)]
    return processor_cls(
        image_processor=image_processor,
        tokenizer=tokenizer,
        patch_size=_get_attr(vision_config, "patch_size", default=14),
        vision_feature_select_strategy=_get_attr(config, "vision_feature_select_strategy", default="default"),
        num_additional_image_tokens=_get_attr(vision_config, "num_additional_image_tokens", default=1),
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


__all__ = ["create_tinyllava_processor", "ensure_image_token"]
