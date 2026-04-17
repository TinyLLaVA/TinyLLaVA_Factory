import os
from typing import TypeVar
from collections.abc import Callable
from ...utils import import_modules
from .base import VisionTower

V = TypeVar("V", bound="VisionTower")

VISION_TOWER_FACTORY: dict[str, type[VisionTower]] = {}


def VisionTowerFactory(vision_tower_name: str) -> type[VisionTower]:
    vision_tower_name = vision_tower_name.split(":")[0]
    model: type[VisionTower] | None = None
    for name in VISION_TOWER_FACTORY.keys():
        if name.lower() in vision_tower_name.lower():
            if model is not None:
                raise ValueError(
                    f"Multiple vision towers found for {vision_tower_name}, "
                    "please specify the model name more precisely"
                )
            model = VISION_TOWER_FACTORY[name]
    if not model:
        raise ValueError(f"{vision_tower_name} is not registered")
    return model


def register_vision_tower(name: str) -> Callable[[type[V]], type[V]]:
    def register_vision_tower_cls(cls: type[V]) -> type[V]:
        if name in VISION_TOWER_FACTORY:
            raise ValueError(f"{name} is already registered")
        VISION_TOWER_FACTORY[name] = cls
        return cls

    return register_vision_tower_cls


# automatically import any Python files in the models/ directory
models_dir = os.path.dirname(__file__)
import_modules(models_dir, "tinyllava.model.vision_tower")
