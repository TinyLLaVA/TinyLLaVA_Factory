"""Local module caches retain the namespace independently of import paths."""

from collections import OrderedDict
from types import SimpleNamespace

import pytest

from tinyllava.model.connector.auto.configuration_auto import (
    _LazyConnectorConfigMapping,
)
from tinyllava.model.connector.auto.modeling_auto import _LazyAutoConnectorModelMapping
from tinyllava.model.connector.auto import auto_mappings as connector_mappings
from tinyllava.model.vision_tower.auto import auto_mappings as vision_mappings
from tinyllava.model.llm.auto import auto_mappings
from tinyllava.model.llm.auto.configuration_auto import _LazyLanguageConfigMapping
from tinyllava.model.llm.auto.modeling_auto import _LazyAutoLanguageModelMapping
from tinyllava.model.vision_tower.auto.configuration_auto import (
    _LazyVisionTowerConfigMapping,
)
from tinyllava.model.vision_tower.auto.modeling_auto import (
    _LazyAutoVisionTowerModelMapping,
)


@pytest.mark.parametrize(
    "config_mapping_cls,model_mapping_cls,suffix,package",
    [
        (
            _LazyConnectorConfigMapping,
            _LazyAutoConnectorModelMapping,
            "__tlf_connector",
            "tinyllava.model.connector",
        ),
        (
            _LazyLanguageConfigMapping,
            _LazyAutoLanguageModelMapping,
            "__tlf_language_model",
            "tinyllava.model.llm",
        ),
        (
            _LazyVisionTowerConfigMapping,
            _LazyAutoVisionTowerModelMapping,
            "__tlf_vision_tower",
            "tinyllava.model.vision_tower",
        ),
    ],
)
def test_local_and_native_module_names_have_separate_cache_entries(
    monkeypatch, config_mapping_cls, model_mapping_cls, suffix, package
):
    key = "llama" + suffix

    class LocalConfig:
        pass

    class LocalModel:
        pass

    local_module = SimpleNamespace(LocalConfig=LocalConfig, LocalModel=LocalModel)
    native_module = SimpleNamespace()
    calls = []

    def import_module(name, package=None):
        calls.append((name, package))
        return local_module

    # A new language backend is discovered from the table, without a name branch.
    monkeypatch.setitem(auto_mappings.LANGUAGE_CONFIG_MAPPING_NAMES, key, "LocalConfig")
    monkeypatch.setitem(
        connector_mappings.CONNECTOR_CONFIG_MAPPING_NAMES, key, "LocalConfig"
    )
    monkeypatch.setitem(
        vision_mappings.VISION_TOWER_CONFIG_MAPPING_NAMES, key, "LocalConfig"
    )
    configs = {key: "LocalConfig"}
    config_mapping = config_mapping_cls(configs)
    model_mapping = model_mapping_cls(configs, OrderedDict({key: "LocalModel"}))
    config_mapping._modules["llama"] = native_module
    model_mapping._modules["llama"] = native_module
    monkeypatch.setattr("importlib.import_module", import_module)

    for _ in range(2):
        assert config_mapping[key] is LocalConfig
        assert model_mapping[LocalConfig] is LocalModel
    for mapping in (config_mapping, model_mapping):
        assert mapping._modules["llama"] is native_module
        assert mapping._modules[key] is local_module
    assert calls == [(".llama", package), (".llama", package)]
