import subprocess
import sys
import textwrap
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import (
    CLIPImageProcessor,
    LlavaProcessor,
    PreTrainedTokenizerFast,
    SiglipImageProcessor,
)

from tinyllava.data.image_processor import AutoImageProcessor
from tinyllava.data.processor import AutoProcessor, BaseProcessor
from tinyllava.data.processor.tinyllava import TinyLlavaProcessor
from tinyllava.model import TinyLlavaConfig
from tinyllava.model.vision_tower.mof import MofVisionConfig


def tokenizer():
    backend = Tokenizer(
        WordLevel({"[UNK]": 0, "[PAD]": 1, "<image>": 2, "hello": 3}, unk_token="[UNK]")
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        additional_special_tokens=["<image>"],
    )


@pytest.mark.parametrize("processor_cls", [CLIPImageProcessor, SiglipImageProcessor])
def test_auto_image_processor_preserves_saved_preprocessing(tmp_path, processor_cls):
    if processor_cls is CLIPImageProcessor:
        processor = processor_cls(
            size={"shortest_edge": 16}, crop_size={"height": 16, "width": 16}
        )
    else:
        processor = processor_cls(size={"height": 16, "width": 16})
    # A composite model_type must not override the saved image processor type.
    TinyLlavaConfig().save_pretrained(tmp_path)
    processor.save_pretrained(tmp_path)
    restored = AutoImageProcessor.from_pretrained(tmp_path, local_files_only=True)
    image = Image.fromarray(np.arange(20 * 24 * 3, dtype=np.uint8).reshape(20, 24, 3))
    torch.testing.assert_close(
        restored(image, return_tensors="pt").pixel_values,
        processor(image, return_tensors="pt").pixel_values,
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("strategy,count", [("default", 8), ("full", 10)])
def test_mof_processor_expansion_and_auto_round_trip(tmp_path, strategy, count):
    config = TinyLlavaConfig(
        vision_config=MofVisionConfig(),
        vision_feature_select_strategy=strategy,
        connector_config={"model_type": "mof__tlf_connector"},
    )
    config.vision_config.clip_config.patch_size = 8
    config.vision_config.dinov2_config.patch_size = 8
    image_processor = CLIPImageProcessor(
        size={"shortest_edge": 16}, crop_size={"height": 16, "width": 16}
    )
    processor = BaseProcessor.from_model(
        tokenizer=tokenizer(),
        image_processor=image_processor,
        model=SimpleNamespace(config=config),
    )
    image = Image.new("RGB", (20, 24), "blue")
    inputs = processor(text="<image>", images=image, return_tensors="pt")
    assert inputs.input_ids.eq(processor.image_token_id).sum().item() == count
    assert processor._get_num_multimodal_tokens(
        image_sizes=[(20, 24)]
    ).num_image_tokens == [count]
    assert isinstance(processor, TinyLlavaProcessor)
    processor.save_pretrained(tmp_path)
    config.save_pretrained(tmp_path)
    restored = AutoProcessor.from_pretrained(tmp_path, local_files_only=True)
    assert isinstance(restored, TinyLlavaProcessor)
    assert restored.patch_size == 8
    restored_inputs = restored(text="<image>", images=image, return_tensors="pt")
    torch.testing.assert_close(restored_inputs.input_ids, inputs.input_ids)
    torch.testing.assert_close(restored_inputs.pixel_values, inputs.pixel_values)
    assert "pixel_values" not in restored(text="hello")


def test_standard_models_preserve_hf_llava_token_expansion():
    config = TinyLlavaConfig()
    processor = BaseProcessor.from_model(
        tokenizer=tokenizer(),
        image_processor=CLIPImageProcessor(),
        model=SimpleNamespace(config=config),
    )
    assert type(processor) is TinyLlavaProcessor
    native = LlavaProcessor(
        tokenizer=processor.tokenizer,
        image_processor=processor.image_processor,
        patch_size=processor.patch_size,
        vision_feature_select_strategy=processor.vision_feature_select_strategy,
        num_additional_image_tokens=processor.num_additional_image_tokens,
    )
    image = Image.new("RGB", (224, 224))
    actual = processor(text="<image>", images=image, return_tensors="pt")
    expected = native(text="<image>", images=image, return_tensors="pt")
    torch.testing.assert_close(actual.input_ids, expected.input_ids)
    torch.testing.assert_close(actual.pixel_values, expected.pixel_values)


@pytest.mark.parametrize(
    "connector_type,expected_class",
    [
        ("mlp__tlf_connector", "TinyLlavaProcessor"),
        ("mof__tlf_connector", "TinyLlavaProcessor"),
    ],
)
def test_saved_processor_loads_lazily_without_model_config(
    tmp_path, connector_type, expected_class
):
    config = TinyLlavaConfig(connector_config={"model_type": connector_type})
    processor = BaseProcessor.from_model(
        tokenizer=tokenizer(),
        image_processor=CLIPImageProcessor(),
        model=SimpleNamespace(config=config),
    )
    processor.save_pretrained(tmp_path)
    script = textwrap.dedent("""
        import sys
        from transformers.models.auto.processing_auto import PROCESSOR_MAPPING
        from transformers.models.auto.image_processing_auto import IMAGE_PROCESSOR_MAPPING
        processor_loader = PROCESSOR_MAPPING._load_attr_from_module.__func__
        image_loader = IMAGE_PROCESSOR_MAPPING._load_attr_from_module.__func__
        from tinyllava.data.processor import AutoProcessor
        custom = "tinyllava.data.processor.tinyllava.processing_tinyllava"
        assert custom not in sys.modules
        result = AutoProcessor.from_pretrained(sys.argv[1], local_files_only=True)
        assert type(result).__name__ == sys.argv[2]
        assert custom in sys.modules
        assert PROCESSOR_MAPPING._load_attr_from_module.__func__ is processor_loader
        assert IMAGE_PROCESSOR_MAPPING._load_attr_from_module.__func__ is image_loader
        assert "tinyllava.model.vision_tower.mof.modeling_mof" not in sys.modules
    """)
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path), expected_class],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
