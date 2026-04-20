import os
from typing import TypeVar
from collections.abc import Callable

from .base import Template
from ...utils.import_module import import_modules


T = TypeVar("T", bound=Template)

TEMPLATE_FACTORY: dict[str, type[Template]] = {}


def TemplateFactory(version: str) -> type[Template]:
    template = TEMPLATE_FACTORY.get(version, None)
    if not template:
        raise ValueError(f"{version} is not registered")
    return template


def register_template(name: str) -> Callable[[type[T]], type[T]]:
    def register_template_cls(cls: type[T]) -> type[T]:
        if name in TEMPLATE_FACTORY:
            raise ValueError(f"{name} is already registered")

        TEMPLATE_FACTORY[name] = cls
        return cls

    return register_template_cls


# automatically import any Python files in the models/ directory
models_dir = os.path.dirname(__file__)
import_modules(models_dir, "tinyllava.data.template")
