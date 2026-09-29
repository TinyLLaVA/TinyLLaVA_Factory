"""Custom processors need neither TinyLlava inheritance nor a factory method."""

import json
import sys
from types import ModuleType

from transformers import PreTrainedConfig
from transformers.processing_utils import ProcessorMixin

from tinyllava.data.processor import AutoProcessor
from tinyllava.data.processor.auto import auto_mappings
from tinyllava.data.processor.auto.processing_auto import PROCESSOR_MAPPING


class VideoConfig(PreTrainedConfig):
    model_type = "test_video__tlf_model"


class VideoProcessor(ProcessorMixin):
    attributes = []

    def __init__(self, frame_stride: int = 2):
        self.frame_stride = frame_stride

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        with open(path / "processor_config.json") as handle:
            config = json.load(handle)
        config.pop("processor_class")
        config.update(kwargs)
        return cls(**config)


def test_registered_processor_constructs_and_reloads_without_tinyllava(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(PROCESSOR_MAPPING, "_extra_content", {})
    AutoProcessor.register(VideoConfig, VideoProcessor)
    processor = AutoProcessor.from_config(VideoConfig(), frame_stride=4)
    assert type(processor) is VideoProcessor
    assert processor.frame_stride == 4
    assert not hasattr(VideoProcessor, "from_config")
    (tmp_path / "processor_config.json").write_text(
        json.dumps({"processor_class": "VideoProcessor", "frame_stride": 4})
    )
    restored = AutoProcessor.from_pretrained(tmp_path, frame_stride=8)
    assert type(restored) is VideoProcessor
    assert restored.frame_stride == 8


def test_mapping_paths_are_independent_of_model_and_processor_names(monkeypatch):
    key = VideoConfig.model_type
    config_path = "test_distinct_config_module"
    processor_path = "test_distinct_processing_module"
    config_module = ModuleType(config_path)
    config_module.VideoConfig = VideoConfig
    processor_module = ModuleType(processor_path)
    processor_module.VideoProcessor = VideoProcessor
    monkeypatch.setitem(sys.modules, config_path, config_module)
    monkeypatch.setitem(sys.modules, processor_path, processor_module)
    monkeypatch.setitem(
        auto_mappings.PROCESSOR_CONFIG_MAPPING_NAMES, key, "VideoConfig"
    )
    monkeypatch.setitem(auto_mappings.PROCESSOR_MAPPING_NAMES, key, "VideoProcessor")
    monkeypatch.setitem(auto_mappings.PROCESSOR_CONFIG_MODULE_NAMES, key, config_path)
    monkeypatch.setitem(auto_mappings.PROCESSOR_MODULE_NAMES, key, processor_path)
    monkeypatch.setitem(PROCESSOR_MAPPING._reverse_config_mapping, "VideoConfig", key)
    monkeypatch.setattr(PROCESSOR_MAPPING, "_modules", {})
    assert VideoConfig in PROCESSOR_MAPPING.keys()
    processor = AutoProcessor.from_config(VideoConfig(), frame_stride=3)
    assert type(processor) is VideoProcessor
    assert processor.frame_stride == 3
    assert PROCESSOR_MAPPING._modules[(key, config_path)] is config_module
    assert PROCESSOR_MAPPING._modules[(key, processor_path)] is processor_module
