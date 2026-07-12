from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from transformers import (
    AutoImageProcessor,
    CLIPImageProcessor,
    CLIPVisionModel,
    Dinov2Model,
    SiglipImageProcessor,
    SiglipVisionModel,
)


@dataclass(frozen=True, slots=True)
class VisionTowerSpec:
    model_cls: type
    image_processor_cls: type


VISION_TOWER_REGISTRY: dict[str, VisionTowerSpec] = {
    "clip": VisionTowerSpec(CLIPVisionModel, CLIPImageProcessor),
    "clip_vision_model": VisionTowerSpec(CLIPVisionModel, CLIPImageProcessor),
    "dinov2": VisionTowerSpec(Dinov2Model, AutoImageProcessor),
    "siglip": VisionTowerSpec(SiglipVisionModel, SiglipImageProcessor),
    "siglip_vision_model": VisionTowerSpec(SiglipVisionModel, SiglipImageProcessor),
}


def get_vision_tower_spec(model_type: str) -> VisionTowerSpec:
    try:
        return VISION_TOWER_REGISTRY[model_type]
    except KeyError as exc:
        supported = ", ".join(sorted(VISION_TOWER_REGISTRY))
        raise ValueError(f"Unsupported vision tower type {model_type!r}. Supported types: {supported}.") from exc


def create_vision_model_from_config(config: Any):
    return get_vision_tower_spec(config.model_type).model_cls(config)


def load_vision_model(model_name_or_path: str, model_type: str | None = None, **kwargs):
    if model_type is None:
        from transformers import AutoConfig

        model_type = AutoConfig.from_pretrained(model_name_or_path).model_type
    return get_vision_tower_spec(model_type).model_cls.from_pretrained(model_name_or_path, **kwargs)


def load_image_processor(model_name_or_path: str, model_type: str | None = None, **kwargs):
    """Load the image processor for known VT types, or defer to HF AutoImageProcessor.

    The vision model itself is resolved by `AutoVisionTowerModel`, including
    TinyLLaVA custom entries registered through the lazy auto mapping. This
    table only captures defaults for common HF vision towers whose processor
    class is known upfront.
    """
    if model_type is None:
        from transformers import AutoConfig

        model_type = AutoConfig.from_pretrained(model_name_or_path).model_type
    spec = VISION_TOWER_REGISTRY.get(model_type)
    image_processor_cls = spec.image_processor_cls if spec is not None else AutoImageProcessor
    return image_processor_cls.from_pretrained(model_name_or_path, **kwargs)


__all__ = [
    "VISION_TOWER_REGISTRY",
    "VisionTowerSpec",
    "create_vision_model_from_config",
    "get_vision_tower_spec",
    "load_image_processor",
    "load_vision_model",
]
