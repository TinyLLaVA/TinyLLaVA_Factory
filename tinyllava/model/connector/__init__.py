import os
from typing import Callable, TypeVar
from ...utils import import_modules
from .base import Connector

C = TypeVar("C", bound=Connector)

CONNECTOR_FACTORY: dict[str, type[Connector]] = {}


def ConnectorFactory(connector_name):
    model = None
    for name in CONNECTOR_FACTORY.keys():
        if name.lower() in connector_name.lower():
            model = CONNECTOR_FACTORY[name]
    assert model, f"{connector_name} is not registered"
    return model


def register_connector(name: str) -> Callable[[type[C]], type[C]]:
    def register_connector_cls(cls: type[C]) -> type[C]:
        if name in CONNECTOR_FACTORY:
            raise ValueError(f"{name} is already registered")
        CONNECTOR_FACTORY[name] = cls
        return cls

    return register_connector_cls


# automatically import any Python files in the models/ directory
models_dir = os.path.dirname(__file__)
import_modules(models_dir, "tinyllava.model.connector")
