import json
import sys

from transformers import CLIPImageProcessor
from transformers.models.auto.image_processing_auto import IMAGE_PROCESSOR_MAPPING
from transformers.utils.import_utils import _LazyModule

import tinyllava.data.image_processor as image_processors
from tinyllava.data.image_processor import AutoImageProcessor
from tinyllava.data.image_processor.auto.auto_mappings import (
    CUSTOM_IMAGE_PROCESSOR_MAPPING_NAMES,
)


def test_local_image_processor_is_loaded_only_when_selected(tmp_path, monkeypatch):
    loader = IMAGE_PROCESSOR_MAPPING._load_attr_from_module.__func__
    names = dict(IMAGE_PROCESSOR_MAPPING._model_mapping)
    assert isinstance(image_processors, _LazyModule)
    extensions = tmp_path / "extensions"
    extensions.mkdir()
    (extensions / "test_extension.py").write_text(
        "from transformers import CLIPImageProcessor\n"
        "class ExtensionImageProcessor(CLIPImageProcessor):\n"
        "    pass\n"
    )
    monkeypatch.setattr(
        image_processors, "__path__", [*image_processors.__path__, str(extensions)]
    )
    monkeypatch.setitem(
        CUSTOM_IMAGE_PROCESSOR_MAPPING_NAMES,
        "test_extension__tlf_image_processor",
        "ExtensionImageProcessor",
    )
    module_name = "tinyllava.data.image_processor.test_extension"
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    native_path = tmp_path / "native"
    CLIPImageProcessor().save_pretrained(native_path)

    native = AutoImageProcessor.from_pretrained(native_path, local_files_only=True)
    assert isinstance(native, CLIPImageProcessor)
    assert module_name not in sys.modules

    custom_path = tmp_path / "custom"
    custom_path.mkdir()
    config = CLIPImageProcessor().to_dict()
    config["image_processor_type"] = "ExtensionImageProcessor"
    (custom_path / "preprocessor_config.json").write_text(json.dumps(config))
    custom = AutoImageProcessor.from_pretrained(
        custom_path, local_files_only=True, do_resize=False
    )
    assert type(custom).__name__ == "ExtensionImageProcessor"
    assert module_name in sys.modules
    assert custom.do_resize is False
    custom.save_pretrained(tmp_path / "saved")
    restored = AutoImageProcessor.from_pretrained(
        tmp_path / "saved", local_files_only=True
    )
    assert type(restored) is type(custom)
    assert restored.do_resize is False
    assert IMAGE_PROCESSOR_MAPPING._load_attr_from_module.__func__ is loader
    assert dict(IMAGE_PROCESSOR_MAPPING._model_mapping) == names
