import subprocess
import sys
from types import SimpleNamespace

import pytest
from PIL import Image
from transformers import CLIPImageProcessor

from tests.data.test_auto_vision_processing import tokenizer
from tinyllava.data.processor import AutoProcessor, create_tinyllava_processor
from tinyllava.model import TinyLlavaConfig


@pytest.mark.parametrize("name", ["qformer", "resampler"])
def test_query_processor_matches_connector_and_saves_without_model_config(
    tmp_path, name
):
    config = TinyLlavaConfig(
        connector_config={"model_type": name + "__tlf_connector", "num_queries": 3}
    )
    processor = create_tinyllava_processor(
        tokenizer=tokenizer(),
        image_processor=CLIPImageProcessor(
            size={"shortest_edge": 16}, crop_size={"height": 16, "width": 16}
        ),
        model=SimpleNamespace(config=config),
    )
    image = Image.new("RGB", (20, 24))
    inputs = processor(
        text="<image><image>", images=[image, image], return_tensors="pt"
    )
    assert inputs.input_ids.eq(processor.image_token_id).sum().item() == 6
    assert processor._get_num_multimodal_tokens(
        image_sizes=[(20, 24)]
    ).num_image_tokens == [3]
    processor.save_pretrained(tmp_path)
    restored = AutoProcessor.from_pretrained(tmp_path, local_files_only=True)
    assert restored.connector_config.num_queries == 3
    assert restored.patch_size == processor.patch_size
    assert (
        restored(text="<image>", images=image, return_tensors="pt")
        .input_ids.eq(restored.image_token_id)
        .sum()
        .item()
        == 3
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from tinyllava.data.processor import AutoProcessor
p = AutoProcessor.from_pretrained(sys.argv[1], local_files_only=True)
assert p.connector_config.num_queries == 3
assert 'tinyllava.model.connector.qformer.modeling_qformer' not in sys.modules
assert 'tinyllava.model.connector.resampler.modeling_resampler' not in sys.modules
""",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
