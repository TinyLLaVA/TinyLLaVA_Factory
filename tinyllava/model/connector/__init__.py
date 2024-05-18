import os

from ...utils import import_modules


CONNECTOR_FACTORY = {}

def ConnectorFactory(connector_name):
    model = None
    for name in CONNECTOR_FACTORY.keys():
        if name.lower() in connector_name.lower():
            model = CONNECTOR_FACTORY[name]
    assert model, f"{connector_name} is not registered"
    return model


def register_connector(name):
    def register_connector_cls(cls):
        if name in CONNECTOR_FACTORY:
            return CONNECTOR_FACTORY[name]
        CONNECTOR_FACTORY[name] = cls
        return cls
    return register_connector_cls


# automatically import any Python files in the models/ directory
models_dir = os.path.dirname(__file__)
import_modules(models_dir, "tinyllava.model.connector")
