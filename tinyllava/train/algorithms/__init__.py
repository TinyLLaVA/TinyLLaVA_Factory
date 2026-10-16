"""Implemented algorithms; importing this module does not load a trainer."""

from importlib import import_module
from types import ModuleType

_ALGORITHMS = {"sft": "tinyllava.train.algorithms.sft"}


def get_algorithm(name: str) -> ModuleType:
    """Return an algorithm owning its sections, validation, arguments, and execution."""
    if not isinstance(name, str) or name not in _ALGORITHMS:
        raise ValueError(
            f"Unsupported training algorithm {name!r}. Implemented algorithms: "
            + ", ".join(_ALGORITHMS)
        )
    return import_module(_ALGORITHMS[name])
